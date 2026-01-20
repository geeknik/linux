// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Synaptic Memory Network (SMN) - Neuron Management
 *
 * Copyright (C) 2026
 *
 * Neuron creation, activation, and lookup functions.
 */

#define pr_fmt(fmt) "smn: neuron: " fmt

#include <linux/slab.h>
#include <linux/gfp.h>
#include <linux/mm.h>
#include <linux/radix-tree.h>
#include <linux/vmalloc.h>

#include "synaptic.h"

/*
 * Create a new neuron
 */
struct synaptic_neuron *smn_neuron_create(enum neuron_type type, void *id)
{
	struct synaptic_neuron *neuron;

	neuron = kzalloc(sizeof(*neuron), GFP_KERNEL);
	if (!neuron)
		return ERR_PTR(-ENOMEM);

	neuron->type = type;
	neuron->id.raw = id;
	neuron->last_activation = smn_time_ms();
	neuron->prev_activation = neuron->last_activation;
	neuron->last_activation_ns = smn_time_ns();
	neuron->prev_activation_ns = neuron->last_activation_ns;
	neuron->activation_count = 0;
	neuron->activation_rate = 0;

	/* Allocate outgoing synapses array */
	neuron->out_capacity = SMN_MAX_SYNAPSES_PER_NEURON;
	neuron->outgoing = kzalloc(
		sizeof(*neuron->outgoing) * neuron->out_capacity, GFP_KERNEL);
	if (!neuron->outgoing) {
		kfree(neuron);
		return ERR_PTR(-ENOMEM);
	}
	neuron->out_count = 0;

	/* Allocate incoming synapses array (pointers) */
	neuron->in_capacity = SMN_MAX_SYNAPSES_PER_NEURON;
	neuron->incoming = kzalloc(
		sizeof(*neuron->incoming) * neuron->in_capacity, GFP_KERNEL);
	if (!neuron->incoming) {
		kfree(neuron->outgoing);
		kfree(neuron);
		return ERR_PTR(-ENOMEM);
	}
	neuron->in_count = 0;

	/* Position info */
	neuron->layer = NULL;
	neuron->nid = NUMA_NO_NODE;

	/* Flags */
	neuron->flags = 0;

	/* Stats */
	memset(&neuron->stats, 0, sizeof(neuron->stats));

	spin_lock_init(&neuron->lock);
	INIT_LIST_HEAD(&neuron->list);
	INIT_LIST_HEAD(&neuron->recent);

	SMN_DBG("Created neuron %p (type=%d, id=%p)\n", neuron, type, id);

	return neuron;
}

/*
 * Destroy a neuron and free its resources
 */
void smn_neuron_destroy(struct synaptic_neuron *neuron)
{
	if (!neuron)
		return;

	SMN_DBG("Destroying neuron %p\n", neuron);

	/* Remove from layer's list */
	if (neuron->layer) {
		list_del(&neuron->list);
		neuron->layer->neuron_count--;
	}

	/* Free outgoing synapses */
	if (neuron->outgoing) {
		/* Note: we don't recursively destroy destination neurons */
		kfree(neuron->outgoing);
	}

	/* Free incoming synapses array */
	if (neuron->incoming) {
		kfree(neuron->incoming);
	}

	kfree(neuron);
}

/*
 * Activate a neuron (called on page access)
 */
void smn_neuron_activate(struct synaptic_neuron *neuron)
{
	unsigned long flags;
	u64 now_ms;
	u64 now_ns;
	u64 interval;

	if (!neuron)
		return;

	now_ms = smn_time_ms();
	now_ns = smn_time_ns();

	spin_lock_irqsave(&neuron->lock, flags);

	/* Update activation state - save previous BEFORE overwriting */
	neuron->prev_activation = neuron->last_activation;
	neuron->prev_activation_ns = neuron->last_activation_ns;
	neuron->last_activation = now_ms;
	neuron->last_activation_ns = now_ns;
	neuron->activation_count++;

	if (neuron->prev_activation > 0) {
		interval = now_ms - neuron->prev_activation;
		if (interval > 0) {
			u16 new_rate = 1000 / interval;
			neuron->activation_rate =
				(neuron->activation_rate * 7 + new_rate) / 8;
		}
	}

	neuron->flags |= NEURON_FLAG_ACTIVE;
	neuron->stats.total_activations++;

	spin_unlock_irqrestore(&neuron->lock, flags);

	if (neuron->layer) {
		struct synaptic_layer *layer = neuron->layer;
		unsigned long rflags;

		layer->stats.total_activations++;

		spin_lock_irqsave(&layer->recent_lock, rflags);
		if (!list_empty(&neuron->recent))
			list_del_init(&neuron->recent);
		list_add(&neuron->recent, &layer->recent_list);
		if (layer->recent_count < 256)
			layer->recent_count++;
		spin_unlock_irqrestore(&layer->recent_lock, rflags);
	}

	SMN_DBG_NEURON(neuron, "activated (count=%u, rate=%u Hz)\n",
		       neuron->activation_count, neuron->activation_rate);

	smn_prefetch_from_neuron(neuron);
}

/*
 * Find neuron by page address
 */
struct synaptic_neuron *smn_page_to_neuron(struct page *page)
{
	struct synaptic_layer *layer;
	unsigned long addr;

	if (!page)
		return NULL;

	/* Use page layer for page-level lookups */
	layer = smn_global.layers[LAYER_PAGE];
	if (!layer)
		return NULL;

	addr = page_to_pfn(page);

	return smn_layer_find_neuron(layer, addr);
}

/*
 * Get page from neuron
 */
struct page *smn_neuron_to_page(struct synaptic_neuron *neuron)
{
	if (!neuron || neuron->type != NEURON_PAGE)
		return NULL;

	return neuron->id.page;
}

static void smn_batch_flush(struct synaptic_layer *layer)
{
	int i, count;
	struct synaptic_neuron *neurons[SMN_BATCH_SIZE];
	unsigned long flags;

	spin_lock_irqsave(&layer->recent_lock, flags);
	count = atomic_read(&layer->batch_count);
	if (count == 0) {
		spin_unlock_irqrestore(&layer->recent_lock, flags);
		return;
	}

	if (count > SMN_BATCH_SIZE)
		count = SMN_BATCH_SIZE;

	for (i = 0; i < count; i++) {
		neurons[i] = layer->batch_queue[i];
		layer->batch_queue[i] = NULL;
	}
	atomic_set(&layer->batch_count, 0);
	layer->batch_start_time = smn_time_ms();
	spin_unlock_irqrestore(&layer->recent_lock, flags);

	for (i = 0; i < count; i++) {
		if (neurons[i])
			smn_learn_from_recent(neurons[i]);
	}
}

void smn_batch_work_func(struct work_struct *work)
{
	struct synaptic_layer *layer =
		container_of(work, struct synaptic_layer, batch_work);
	smn_batch_flush(layer);
}

static void smn_batch_add(struct synaptic_layer *layer,
			  struct synaptic_neuron *neuron)
{
	int idx;
	u64 now;
	unsigned long flags;

	spin_lock_irqsave(&layer->recent_lock, flags);
	idx = atomic_fetch_add(1, &layer->batch_count);
	if (idx < SMN_BATCH_SIZE)
		layer->batch_queue[idx] = neuron;
	spin_unlock_irqrestore(&layer->recent_lock, flags);

	if (idx + 1 >= SMN_BATCH_SIZE) {
		if (smn_global.stats_wq)
			queue_work(smn_global.stats_wq, &layer->batch_work);
		return;
	}

	now = smn_time_ms();
	if (now - layer->batch_start_time >= SMN_BATCH_INTERVAL_MS) {
		if (smn_global.stats_wq)
			queue_work(smn_global.stats_wq, &layer->batch_work);
	}
}

void smn_mark_page_accessed(struct page *page)
{
	struct synaptic_neuron *neuron;
	struct synaptic_layer *layer;
	int sample;

	if (!smn_global.initialized || !page)
		return;

	layer = smn_global.layers[LAYER_PAGE];
	if (!layer)
		return;

	sample = atomic_fetch_add(1, &layer->sample_counter);
	if ((sample & (SMN_SAMPLE_RATE - 1)) != 0) {
		layer->stats.total_activations++;
		return;
	}

	neuron = smn_layer_find_neuron(layer, page_to_pfn(page));

	if (!neuron) {
		neuron = smn_neuron_create(NEURON_PAGE, page);
		if (IS_ERR(neuron))
			return;

		neuron->layer = layer;
		neuron->nid = page_to_nid(page);
		smn_layer_add_neuron(layer, neuron);
	}

	smn_neuron_activate(neuron);
	smn_batch_add(layer, neuron);
}

/*
 * Check if a neuron should be reclaimed
 */
bool smn_should_reclaim(struct synaptic_neuron *neuron)
{
	unsigned long flags;
	u64 idle_time;
	u16 total_weight = 0;
	int i;

	if (!neuron)
		return true;

	spin_lock_irqsave(&neuron->lock, flags);

	/* Time-based protection (like LRU) */
	idle_time = smn_time_ms() - neuron->last_activation;
	if (idle_time < SMN_RECLAIM_IDLE_THRESHOLD) {
		spin_unlock_irqrestore(&neuron->lock, flags);
		return false;
	}

	/* Relationship-based protection (SMN novelty) */
	/* Sum all outgoing synapse weights */
	for (i = 0; i < neuron->out_count; i++) {
		total_weight += neuron->outgoing[i].weight;
	}

	if (total_weight > smn_global.config.reclaim_protection) {
		spin_unlock_irqrestore(&neuron->lock, flags);
		return false;
	}

	/* Frequency-based protection */
	if (neuron->activation_rate > SMN_ACTIVE_RATE_THRESHOLD) {
		spin_unlock_irqrestore(&neuron->lock, flags);
		return false;
	}

	spin_unlock_irqrestore(&neuron->lock, flags);

	return true;
}

/*
 * Check if page should be reclaimed
 */
bool smn_page_should_reclaim(struct page *page)
{
	struct synaptic_neuron *neuron;

	if (!smn_global.initialized)
		return false;

	neuron = smn_page_to_neuron(page);
	if (!neuron)
		return true; /* No neuron = not protected */

	return smn_should_reclaim(neuron);
}

/*
 * Get neuron statistics
 */
void smn_dump_neurons(struct synaptic_layer *layer)
{
	struct synaptic_neuron *neuron;
	int count = 0;

	pr_info("SMN Neurons in layer %p:\n", layer);
	pr_info("  Total: %u\n", layer->neuron_count);

	list_for_each_entry(neuron, &layer->neuron_list, list) {
		pr_info("  [%d] Neuron %p: type=%d, activations=%u, rate=%u Hz\n",
			count++, neuron, neuron->type, neuron->activation_count,
			neuron->activation_rate);
		pr_info("       Synapses: out=%u, in=%u\n", neuron->out_count,
			neuron->in_count);

		/* Only print first 20 to avoid spam */
		if (count >= 20) {
			pr_info("  ... and %u more\n",
				layer->neuron_count - 20);
			break;
		}
	}
}
