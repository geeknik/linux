// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Synaptic Memory Network (SMN) - Layer Management
 *
 * Copyright (C) 2026
 *
 * Layer creation, management, and neuron lookup.
 */

#define pr_fmt(fmt) "smn: layer: " fmt

#include <linux/slab.h>
#include <linux/gfp.h>
#include <linux/kernel.h>
#include <linux/radix-tree.h>
#include <linux/workqueue.h>
#include <linux/delay.h>

#include "synaptic.h"

/*
 * Create a new synaptic layer
 */
struct synaptic_layer *smn_layer_create(enum layer_type type)
{
	struct synaptic_layer *layer;

	layer = kzalloc(sizeof(*layer), GFP_KERNEL);
	if (!layer)
		return ERR_PTR(-ENOMEM);

	layer->type = type;
	layer->neuron_capacity = SMN_NEURON_INITIAL_CAPACITY;
	layer->neuron_count = 0;

	/* Allocate neuron array */
	layer->neurons = kzalloc(sizeof(*layer->neurons) * layer->neuron_capacity,
				 GFP_KERNEL);
	if (!layer->neurons) {
		kfree(layer);
		return ERR_PTR(-ENOMEM);
	}

	/* Initialize radix tree for fast lookup */
	INIT_RADIX_TREE(&layer->neuron_tree, GFP_ATOMIC);

	/* No previous/next layers initially */
	layer->prev_layer = NULL;
	layer->next_layer = NULL;
	layer->inter_synapses = NULL;

	/* Initialize stats */
	memset(&layer->stats, 0, sizeof(layer->stats));

	/* Initialize lock and list */
	mutex_init(&layer->lock);
	INIT_LIST_HEAD(&layer->neuron_list);

	/* Initialize prune work */
	INIT_DELAYED_WORK(&layer->prune_work, NULL); /* Will be set later */

	SMN_DBG("Created layer %p (type=%d)\n", layer, type);

	return layer;
}

/*
 * Destroy a synaptic layer
 */
void smn_layer_destroy(struct synaptic_layer *layer)
{
	struct synaptic_neuron *neuron, *tmp;

	if (!layer)
		return;

	SMN_DBG("Destroying layer %p\n", layer);

	/* Cancel prune work */
	cancel_delayed_work_sync(&layer->prune_work);

	/* Destroy all neurons */
	list_for_each_entry_safe(neuron, tmp, &layer->neuron_list, list) {
		smn_neuron_destroy(neuron);
	}

	/* Free neuron array */
	if (layer->neurons)
		kfree(layer->neurons);

	/* Free inter-layer synapses */
	if (layer->inter_synapses)
		kfree(layer->inter_synapses);

	kfree(layer);
}

/*
 * Add a neuron to a layer
 */
int smn_layer_add_neuron(struct synaptic_layer *layer,
			 struct synaptic_neuron *neuron)
{
	unsigned long addr = 0;
	int ret;

	if (!layer || !neuron)
		return -EINVAL;

	mutex_lock(&layer->lock);

	/* Check capacity */
	if (layer->neuron_count >= layer->neuron_capacity) {
		/* Expand capacity */
		u32 new_capacity = layer->neuron_capacity * 2;
		struct synaptic_neuron **new_neurons;

		new_neurons = krealloc(layer->neurons,
				      sizeof(*new_neurons) * new_capacity,
				      GFP_KERNEL);
		if (!new_neurons) {
			mutex_unlock(&layer->lock);
			return -ENOMEM;
		}

		layer->neurons = new_neurons;
		layer->neuron_capacity = new_capacity;
	}

	/* Add to array */
	layer->neurons[layer->neuron_count] = neuron;
	layer->neuron_count++;

	/* Add to radix tree for lookup */
	switch (neuron->type) {
	case NEURON_PAGE:
		addr = page_to_pfn(neuron->id.page);
		break;
	case NEURON_VMA:
		addr = (unsigned long)neuron->id.vma->vm_start;
		break;
	default:
		addr = (unsigned long)neuron->id.raw;
		break;
	}

	ret = radix_tree_insert(&layer->neuron_tree, addr, neuron);
	if (ret) {
		/* Rollback */
		layer->neuron_count--;
		mutex_unlock(&layer->lock);
		return ret;
	}

	/* Add to list */
	list_add_tail(&neuron->list, &layer->neuron_list);

	/* Link neuron to layer */
	neuron->layer = layer;

	mutex_unlock(&layer->lock);

	SMN_DBG("Added neuron %p to layer %p (addr=%lx)\n",
		neuron, layer, addr);

	return 0;
}

/*
 * Find a neuron by address in a layer
 */
struct synaptic_neuron *smn_layer_find_neuron(struct synaptic_layer *layer,
					      unsigned long addr)
{
	if (!layer)
		return NULL;

	return radix_tree_lookup(&layer->neuron_tree, addr);
}

/*
 * Prune weak synapses in a layer
 */
void smn_prune_synapses(struct synaptic_layer *layer)
{
	struct synaptic_neuron *neuron;
	int i, j;
	unsigned long flags;

	if (!layer)
		return;

	SMN_DBG("Pruning layer %p\n", layer);

	list_for_each_entry(neuron, &layer->neuron_list, list) {
		spin_lock_irqsave(&neuron->lock, flags);

		for (i = 0; i < neuron->out_count; ) {
			struct synapse *s = &neuron->outgoing[i];
			u64 idle_time = smn_time_ms() - s->last_activated;
			u16 decay;

			/* Decay based on inactivity */
			decay = idle_time / smn_global.config.decay_rate;

			/* Strong synapses resist decay */
			if (s->flags & SYNAPSE_FLAG_BIDIRECTIONAL)
				decay /= 2;

			/* Apply decay */
			if (s->weight > decay) {
				s->weight -= decay;
			} else {
				s->weight = 0;
			}

			/* Remove very weak synapses */
			if (s->weight < SMN_MIN_WEIGHT && s->weight == 0) {
				/* Remove by shifting */
				for (j = i; j < neuron->out_count - 1; j++) {
					neuron->outgoing[j] = neuron->outgoing[j + 1];
				}
				neuron->out_count--;
				/* Don't increment i, check new element at this position */
				continue;
			}

			i++;
		}

		spin_unlock_irqrestore(&neuron->lock, flags);
	}
}

/*
 * Decay all synapses (called periodically)
 */
void smn_decay_synapses(struct synaptic_layer *layer)
{
	struct synaptic_neuron *neuron;
	int i;
	u64 now = smn_time_ms();

	if (!layer)
		return;

	list_for_each_entry(neuron, &layer->neuron_list, list) {
		for (i = 0; i < neuron->out_count; i++) {
			struct synapse *s = &neuron->outgoing[i];
			u64 idle = now - s->last_activated;
			u16 decay = idle / smn_global.config.decay_rate;

			if (decay > 0 && s->weight > 0) {
				/* Strong synapses resist decay */
				if (s->flags & SYNAPSE_FLAG_BIDIRECTIONAL)
					decay /= 2;

				s->weight = max(0, s->weight - decay);
			}
		}
	}
}

/*
 * Prune work function
 */
static void __always_unused smn_prune_work_func(struct work_struct *work)
{
	struct synaptic_layer *layer;

	layer = container_of(to_delayed_work(work),
			     struct synaptic_layer, prune_work);

	smn_prune_synapses(layer);

	/* Reschedule */
	queue_delayed_work(smn_global.prune_wq, &layer->prune_work,
			   msecs_to_jiffies(smn_global.config.prune_interval));
}

/*
 * Start pruning for a layer
 */
void smn_layer_start_pruning(struct synaptic_layer *layer)
{
	if (!layer)
		return;

	queue_delayed_work(smn_global.prune_wq, &layer->prune_work,
			   msecs_to_jiffies(smn_global.config.prune_interval));
}
