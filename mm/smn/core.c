// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Synaptic Memory Network (SMN) - Core Subsystem
 *
 * Copyright (C) 2026
 *
 * Core initialization and management for the bio-inspired
 * memory management system.
 */

#define pr_fmt(fmt) "smn: " fmt

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/workqueue.h>
#include <linux/sysfs.h>
#include <linux/kobject.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/delay.h>
#include <linux/debugfs.h>

#include "synaptic.h"

/*
 * Global SMN system instance
 */
struct smn_system smn_global = {
	.initialized = false,
};

/*
 * Default configuration
 */
static const struct smn_config smn_config_default = {
	.learning_mode = SMN_LEARNING_HEBBIAN,
	.learning_rate = SMN_HEBBIAN_LEARNING_RATE,
	.decay_rate = SMN_DECAY_RATE / 1000, /* Convert to seconds */
	.coactivation_window = SMN_COACTIVATION_WINDOW / NSEC_PER_MSEC,
	.predictive_threshold = SMN_PREDICTIVE_THRESHOLD,
	.colocate_threshold = SMN_COLOCATE_THRESHOLD,
	.reclaim_protection = SMN_RECLAIM_PROTECT_THRESHOLD,
	.max_synapses_per_neuron = SMN_MAX_SYNAPSES_PER_NEURON,
	.prune_interval = 60000, /* 60 seconds */
	.enable_prefetch = true,
	.enable_colocate = true,
};

/*
 * Module parameters
 */
static int learning_mode = SMN_LEARNING_HEBBIAN;
module_param(learning_mode, int, 0644);
MODULE_PARM_DESC(learning_mode,
		 "Learning algorithm (0=none, 1=hebbian, 2=stdp, 3=hybrid)");

static int enable_prefetch = 1;
module_param(enable_prefetch, int, 0644);
MODULE_PARM_DESC(enable_prefetch, "Enable synaptic prefetching");

static int enable_colocate = 1;
module_param(enable_colocate, int, 0644);
MODULE_PARM_DESC(enable_colocate, "Enable synaptic co-location");

static int max_synapses = SMN_MAX_SYNAPSES_PER_NEURON;
module_param(max_synapses, int, 0644);
MODULE_PARM_DESC(max_synapses, "Maximum synapses per neuron");

static int prune_interval = 60000;
module_param(prune_interval, int, 0644);
MODULE_PARM_DESC(prune_interval, "Pruning interval in milliseconds");

/*
 * Initialize the SMN system
 */
int __init smn_init(void)
{
	int ret;
	int i;

	pr_info("Initializing Synaptic Memory Network\n");

	if (smn_global.initialized) {
		pr_warn("Already initialized\n");
		return 0;
	}

	/* Initialize configuration */
	smn_global.config = smn_config_default;

	/* Override with module parameters */
	smn_global.config.learning_mode = learning_mode;
	smn_global.config.enable_prefetch = enable_prefetch;
	smn_global.config.enable_colocate = enable_colocate;
	smn_global.config.max_synapses_per_neuron = max_synapses;
	smn_global.config.prune_interval = prune_interval;

	mutex_init(&smn_global.lock);
	hash_init(smn_global.mm_hash);
	spin_lock_init(&smn_global.mm_hash_lock);

	/* Create workqueues */
	smn_global.stats_wq =
		alloc_workqueue("smn_stats", WQ_MEM_RECLAIM | WQ_UNBOUND, 0);
	if (!smn_global.stats_wq) {
		pr_err("Failed to allocate stats workqueue\n");
		ret = -ENOMEM;
		goto err_wq_stats;
	}

	smn_global.prune_wq =
		alloc_workqueue("smn_prune", WQ_MEM_RECLAIM | WQ_UNBOUND, 0);
	if (!smn_global.prune_wq) {
		pr_err("Failed to allocate prune workqueue\n");
		ret = -ENOMEM;
		goto err_wq_prune;
	}

	/* Create synaptic layers */
	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		smn_global.layers[i] = smn_layer_create(i);
		if (IS_ERR(smn_global.layers[i])) {
			ret = PTR_ERR(smn_global.layers[i]);
			pr_err("Failed to create layer %d: %d\n", i, ret);
			goto err_layers;
		}
	}

	smn_global.initialized = true;

	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		if (smn_global.layers[i])
			smn_layer_start_pruning(smn_global.layers[i]);
	}

	pr_info("SMN initialized successfully\n");
	pr_info("  Learning mode: %s\n",
		smn_global.config.learning_mode == SMN_LEARNING_HEBBIAN ?
			"Hebbian" :
		smn_global.config.learning_mode == SMN_LEARNING_STDP ?
			"STDP" :
		smn_global.config.learning_mode == SMN_LEARNING_HYBRID ?
			"Hybrid" :
			"None");
	pr_info("  Prefetch: %s\n",
		smn_global.config.enable_prefetch ? "enabled" : "disabled");
	pr_info("  Co-location: %s\n",
		smn_global.config.enable_colocate ? "enabled" : "disabled");
	pr_info("  Max synapses/neuron: %d\n",
		smn_global.config.max_synapses_per_neuron);

	return 0;

err_layers:
	while (--i >= 0) {
		smn_layer_destroy(smn_global.layers[i]);
	}
	destroy_workqueue(smn_global.prune_wq);
err_wq_prune:
	destroy_workqueue(smn_global.stats_wq);
err_wq_stats:
	return ret;
}

/*
 * Cleanup the SMN system
 */
void smn_exit(void)
{
	int i;

	if (!smn_global.initialized)
		return;

	pr_info("Shutting down SMN\n");

	/* Destroy all layers */
	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		if (smn_global.layers[i])
			smn_layer_destroy(smn_global.layers[i]);
	}

	/* Destroy workqueues */
	if (smn_global.stats_wq)
		destroy_workqueue(smn_global.stats_wq);
	if (smn_global.prune_wq)
		destroy_workqueue(smn_global.prune_wq);

	smn_global.initialized = false;

	pr_info("SMN shutdown complete\n");
}

/*
 * Sysfs support
 */

static ssize_t learning_mode_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	return sprintf(buf, "%d\n", smn_global.config.learning_mode);
}

static ssize_t learning_mode_store(struct kobject *kobj,
				   struct kobj_attribute *attr, const char *buf,
				   size_t count)
{
	int mode;

	if (kstrtoint(buf, 10, &mode))
		return -EINVAL;

	if (mode < 0 || mode >= SMN_LEARNING_MAX)
		return -EINVAL;

	smn_global.config.learning_mode = mode;
	return count;
}

static ssize_t enable_prefetch_show(struct kobject *kobj,
				    struct kobj_attribute *attr, char *buf)
{
	return sprintf(buf, "%d\n", smn_global.config.enable_prefetch);
}

static ssize_t enable_prefetch_store(struct kobject *kobj,
				     struct kobj_attribute *attr,
				     const char *buf, size_t count)
{
	bool val;

	if (kstrtobool(buf, &val))
		return -EINVAL;

	smn_global.config.enable_prefetch = val;
	return count;
}

static ssize_t total_neurons_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	u32 total = 0;
	int i;

	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		if (smn_global.layers[i])
			total += smn_global.layers[i]->neuron_count;
	}

	return sprintf(buf, "%u\n", total);
}

static ssize_t total_synapses_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	u64 total = 0;
	int i;

	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		if (smn_global.layers[i])
			total += smn_layer_count_synapses(smn_global.layers[i]);
	}

	return sprintf(buf, "%llu\n", total);
}

static ssize_t enable_colocate_show(struct kobject *kobj,
				    struct kobj_attribute *attr, char *buf)
{
	return sprintf(buf, "%d\n", smn_global.config.enable_colocate);
}

static ssize_t enable_colocate_store(struct kobject *kobj,
				     struct kobj_attribute *attr,
				     const char *buf, size_t count)
{
	bool val;

	if (kstrtobool(buf, &val))
		return -EINVAL;

	smn_global.config.enable_colocate = val;
	return count;
}

static ssize_t learning_rate_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	return sprintf(buf, "%u\n", smn_global.config.learning_rate);
}

static ssize_t learning_rate_store(struct kobject *kobj,
				   struct kobj_attribute *attr, const char *buf,
				   size_t count)
{
	u16 val;

	if (kstrtou16(buf, 10, &val))
		return -EINVAL;

	if (val > 100)
		return -EINVAL;

	smn_global.config.learning_rate = val;
	return count;
}

static ssize_t predictive_threshold_show(struct kobject *kobj,
					 struct kobj_attribute *attr, char *buf)
{
	return sprintf(buf, "%u\n", smn_global.config.predictive_threshold);
}

static ssize_t predictive_threshold_store(struct kobject *kobj,
					  struct kobj_attribute *attr,
					  const char *buf, size_t count)
{
	u16 val;

	if (kstrtou16(buf, 10, &val))
		return -EINVAL;

	if (val > SMN_WEIGHT_MAX)
		return -EINVAL;

	smn_global.config.predictive_threshold = val;
	return count;
}

static ssize_t total_activations_show(struct kobject *kobj,
				      struct kobj_attribute *attr, char *buf)
{
	u64 total = 0;
	int i;

	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		if (smn_global.layers[i])
			total += smn_global.layers[i]->stats.total_activations;
	}

	return sprintf(buf, "%llu\n", total);
}

static struct kobj_attribute smn_attr_learning_mode =
	__ATTR(learning_mode, 0644, learning_mode_show, learning_mode_store);

static struct kobj_attribute smn_attr_enable_prefetch = __ATTR(
	enable_prefetch, 0644, enable_prefetch_show, enable_prefetch_store);

static struct kobj_attribute smn_attr_enable_colocate = __ATTR(
	enable_colocate, 0644, enable_colocate_show, enable_colocate_store);

static struct kobj_attribute smn_attr_learning_rate =
	__ATTR(learning_rate, 0644, learning_rate_show, learning_rate_store);

static struct kobj_attribute smn_attr_predictive_threshold =
	__ATTR(predictive_threshold, 0644, predictive_threshold_show,
	       predictive_threshold_store);

static struct kobj_attribute smn_attr_total_neurons =
	__ATTR(total_neurons, 0444, total_neurons_show, NULL);

static struct kobj_attribute smn_attr_total_synapses =
	__ATTR(total_synapses, 0444, total_synapses_show, NULL);

static struct kobj_attribute smn_attr_total_activations =
	__ATTR(total_activations, 0444, total_activations_show, NULL);

static struct attribute *smn_attrs[] = {
	&smn_attr_learning_mode.attr,
	&smn_attr_enable_prefetch.attr,
	&smn_attr_enable_colocate.attr,
	&smn_attr_learning_rate.attr,
	&smn_attr_predictive_threshold.attr,
	&smn_attr_total_neurons.attr,
	&smn_attr_total_synapses.attr,
	&smn_attr_total_activations.attr,
	NULL,
};

static const struct attribute_group smn_attr_group = {
	.attrs = smn_attrs,
};

static struct kobject *smn_kobj;

static int __init smn_sysfs_init(void)
{
	int ret;

	smn_kobj = kobject_create_and_add("smn", kernel_kobj);
	if (!smn_kobj)
		return -ENOMEM;

	ret = sysfs_create_group(smn_kobj, &smn_attr_group);
	if (ret) {
		kobject_put(smn_kobj);
		return ret;
	}

	return 0;
}

static void smn_sysfs_exit(void)
{
	kobject_put(smn_kobj);
}

static struct dentry *smn_debugfs_dir;

static int smn_debugfs_neurons_show(struct seq_file *m, void *v)
{
	struct synaptic_layer *layer;
	struct synaptic_neuron *neuron;
	int i;

	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		layer = smn_global.layers[i];
		if (!layer)
			continue;

		seq_printf(m, "=== Layer %d ===\n", i);

		mutex_lock(&layer->lock);
		list_for_each_entry(neuron, &layer->neuron_list, list) {
			seq_printf(
				m,
				"  Neuron %p: act=%u rate=%u out=%u in=%u flags=0x%x\n",
				neuron->id.raw, neuron->activation_count,
				neuron->activation_rate, neuron->out_count,
				neuron->in_count, neuron->flags);
		}
		mutex_unlock(&layer->lock);
	}

	return 0;
}

static int smn_debugfs_synapses_show(struct seq_file *m, void *v)
{
	struct synaptic_layer *layer;
	struct synaptic_neuron *neuron;
	int j;
	u64 total_synapses = 0;
	u64 total_weight = 0;
	u32 max_weight = 0;
	u32 min_weight = SMN_WEIGHT_MAX;
	u32 predictive_count = 0;
	u32 bidirectional_count = 0;
	u32 colocate_count = 0;
	u32 weight_buckets[11] = { 0 };
	struct {
		void *src;
		void *dst;
		u16 weight;
		u8 flags;
	} top_synapses[10];
	int top_count = 0;

	layer = smn_global.layers[LAYER_PAGE];
	if (!layer)
		return 0;

	mutex_lock(&layer->lock);
	list_for_each_entry(neuron, &layer->neuron_list, list) {
		for (j = 0; j < neuron->out_count; j++) {
			struct synapse *s = &neuron->outgoing[j];
			int bucket;

			total_synapses++;
			total_weight += s->weight;

			if (s->weight > max_weight)
				max_weight = s->weight;
			if (s->weight < min_weight)
				min_weight = s->weight;

			if (s->flags & SYNAPSE_FLAG_PREDICTIVE)
				predictive_count++;
			if (s->flags & SYNAPSE_FLAG_BIDIRECTIONAL)
				bidirectional_count++;
			if (s->flags & SYNAPSE_FLAG_COLOCATE)
				colocate_count++;

			bucket = s->weight / 100;
			if (bucket > 10)
				bucket = 10;
			weight_buckets[bucket]++;

			if (top_count < 10 ||
			    s->weight > top_synapses[9].weight) {
				int pos = top_count < 10 ? top_count : 9;
				int k;

				for (k = pos; k > 0; k--) {
					if (top_synapses[k - 1].weight >=
					    s->weight)
						break;
					if (k < 10)
						top_synapses[k] =
							top_synapses[k - 1];
				}
				top_synapses[k].src = neuron->id.raw;
				top_synapses[k].dst = s->dst ? s->dst->id.raw :
							       NULL;
				top_synapses[k].weight = s->weight;
				top_synapses[k].flags = s->flags;
				if (top_count < 10)
					top_count++;
			}
		}
	}
	mutex_unlock(&layer->lock);

	seq_printf(m, "=== Synapse Summary (Page Layer) ===\n");
	seq_printf(m, "Total synapses: %llu\n", total_synapses);
	if (total_synapses > 0) {
		seq_printf(m, "Average weight: %llu\n",
			   total_weight / total_synapses);
		seq_printf(m, "Min weight: %u\n", min_weight);
		seq_printf(m, "Max weight: %u\n", max_weight);
	}
	seq_printf(m, "\n=== Flag Counts ===\n");
	seq_printf(m, "Predictive (w>%u): %u\n",
		   smn_global.config.predictive_threshold, predictive_count);
	seq_printf(m, "Bidirectional: %u\n", bidirectional_count);
	seq_printf(m, "Colocate (w>%u): %u\n",
		   smn_global.config.colocate_threshold, colocate_count);

	seq_printf(m, "\n=== Weight Distribution ===\n");
	seq_printf(m, "  0-99:   %u\n", weight_buckets[0]);
	seq_printf(m, "100-199:  %u\n", weight_buckets[1]);
	seq_printf(m, "200-299:  %u\n", weight_buckets[2]);
	seq_printf(m, "300-399:  %u\n", weight_buckets[3]);
	seq_printf(m, "400-499:  %u\n", weight_buckets[4]);
	seq_printf(m, "500-599:  %u\n", weight_buckets[5]);
	seq_printf(m, "600-699:  %u\n", weight_buckets[6]);
	seq_printf(m, "700-799:  %u\n", weight_buckets[7]);
	seq_printf(m, "800-899:  %u\n", weight_buckets[8]);
	seq_printf(m, "900-999:  %u\n", weight_buckets[9]);
	seq_printf(m, "1000:     %u\n", weight_buckets[10]);

	seq_printf(m, "\n=== Top %d Strongest Synapses ===\n", top_count);
	for (j = 0; j < top_count; j++) {
		seq_printf(m, "  %p -> %p: w=%u flags=0x%x\n",
			   top_synapses[j].src, top_synapses[j].dst,
			   top_synapses[j].weight, top_synapses[j].flags);
	}

	return 0;
}

static int smn_debugfs_config_show(struct seq_file *m, void *v)
{
	seq_printf(m, "learning_mode: %d\n", smn_global.config.learning_mode);
	seq_printf(m, "learning_rate: %u\n", smn_global.config.learning_rate);
	seq_printf(m, "decay_rate: %u\n", smn_global.config.decay_rate);
	seq_printf(m, "coactivation_window: %u ms\n",
		   smn_global.config.coactivation_window);
	seq_printf(m, "predictive_threshold: %u\n",
		   smn_global.config.predictive_threshold);
	seq_printf(m, "colocate_threshold: %u\n",
		   smn_global.config.colocate_threshold);
	seq_printf(m, "reclaim_protection: %u\n",
		   smn_global.config.reclaim_protection);
	seq_printf(m, "max_synapses_per_neuron: %u\n",
		   smn_global.config.max_synapses_per_neuron);
	seq_printf(m, "prune_interval: %u ms\n",
		   smn_global.config.prune_interval);
	seq_printf(m, "enable_prefetch: %d\n",
		   smn_global.config.enable_prefetch);
	seq_printf(m, "enable_colocate: %d\n",
		   smn_global.config.enable_colocate);

	return 0;
}

DEFINE_SHOW_ATTRIBUTE(smn_debugfs_neurons);
DEFINE_SHOW_ATTRIBUTE(smn_debugfs_synapses);
DEFINE_SHOW_ATTRIBUTE(smn_debugfs_config);

static int __init smn_debugfs_init(void)
{
	smn_debugfs_dir = debugfs_create_dir("smn", NULL);
	if (IS_ERR_OR_NULL(smn_debugfs_dir))
		return -ENOMEM;

	debugfs_create_file("neurons", 0444, smn_debugfs_dir, NULL,
			    &smn_debugfs_neurons_fops);
	debugfs_create_file("synapses", 0444, smn_debugfs_dir, NULL,
			    &smn_debugfs_synapses_fops);
	debugfs_create_file("config", 0444, smn_debugfs_dir, NULL,
			    &smn_debugfs_config_fops);

	return 0;
}

static void smn_debugfs_exit(void)
{
	debugfs_remove_recursive(smn_debugfs_dir);
}

static int smn_proc_show(struct seq_file *m, void *v)
{
	int i;

	seq_printf(m, "Synaptic Memory Network Statistics\n");
	seq_printf(m, "===================================\n\n");
	seq_printf(m, "Learning Mode: %d\n", smn_global.config.learning_mode);
	seq_printf(m, "Prefetch: %s\n",
		   smn_global.config.enable_prefetch ? "enabled" : "disabled");
	seq_printf(m, "Co-location: %s\n",
		   smn_global.config.enable_colocate ? "enabled" : "disabled");
	seq_printf(m, "Max Synapses/Neuron: %d\n",
		   smn_global.config.max_synapses_per_neuron);
	seq_printf(m, "\nLayer Statistics:\n");

	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		struct synaptic_layer *layer = smn_global.layers[i];
		u64 synapse_count;

		if (!layer)
			continue;

		smn_layer_update_stats(layer);
		synapse_count = smn_layer_count_synapses(layer);

		seq_printf(m, "  Layer %d (%s):\n", i,
			   i == LAYER_PAGE   ? "Page" :
			   i == LAYER_VMA    ? "VMA" :
			   i == LAYER_REGION ? "Region" :
					       "Global");
		seq_printf(m, "    Neurons: %u\n", layer->neuron_count);
		seq_printf(m, "    Synapses: %llu\n", synapse_count);
		seq_printf(m, "    Activations: %llu\n",
			   layer->stats.total_activations);
		seq_printf(m, "    Avg Weight: %u\n",
			   layer->stats.avg_synapse_weight);
		seq_printf(m, "    Connectivity: %u\n",
			   layer->stats.connectivity);
	}

	return 0;
}

static int smn_proc_open(struct inode *inode, struct file *file)
{
	return single_open(file, smn_proc_show, NULL);
}

static const struct proc_ops smn_proc_ops = {
	.proc_open = smn_proc_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};

/*
 * Subsystem initialization
 */
static int __init smn_subsys_init(void)
{
	int ret;
	struct proc_dir_entry *entry;

	ret = smn_init();
	if (ret)
		return ret;

	ret = smn_sysfs_init();
	if (ret) {
		pr_err("Failed to create sysfs entries: %d\n", ret);
		goto err_sysfs;
	}

	entry = proc_create("smn_stats", 0444, NULL, &smn_proc_ops);
	if (!entry) {
		pr_err("Failed to create proc entry\n");
		ret = -ENOMEM;
		goto err_proc;
	}

	ret = smn_debugfs_init();
	if (ret)
		pr_warn("debugfs init failed: %d (non-fatal)\n", ret);

	pr_info("SMN subsystem loaded\n");
	return 0;

err_proc:
	smn_sysfs_exit();
err_sysfs:
	smn_exit();
	return ret;
}
module_init(smn_subsys_init);

static void __exit smn_subsys_exit(void)
{
	smn_debugfs_exit();
	remove_proc_entry("smn_stats", NULL);
	smn_sysfs_exit();
	smn_exit();
	pr_info("SMN subsystem unloaded\n");
}
module_exit(smn_subsys_exit);

MODULE_AUTHOR("Anonymous Research Team");
MODULE_DESCRIPTION("Synaptic Memory Network - Bio-inspired MM");
MODULE_LICENSE("GPL");
