// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Synaptic Memory Network (SMN) - Synapse Operations
 *
 * Copyright (C) 2026
 *
 * Synapse creation, management, and operations.
 */

#define pr_fmt(fmt) "smn: synapse: " fmt

#include <linux/slab.h>
#include <linux/gfp.h>
#include <linux/kernel.h>

#include "synaptic.h"

/*
 * Create a new synapse from src to dst
 */
struct synapse *smn_synapse_create(struct synaptic_neuron *src,
				   struct synaptic_neuron *dst)
{
	struct synapse *s;
	unsigned long flags;

	if (!src || !dst)
		return ERR_PTR(-EINVAL);

	/* Check if synapse already exists */
	s = smn_find_synapse(src, dst);
	if (s)
		return s;

	/* Check capacity */
	spin_lock_irqsave(&src->lock, flags);

	if (src->out_count >= src->out_capacity) {
		spin_unlock_irqrestore(&src->lock, flags);
		SMN_DBG("Neuron %p: synapse capacity exceeded\n", src);
		return ERR_PTR(-ENOSPC);
	}

	spin_unlock_irqrestore(&src->lock, flags);

	/* Allocate new synapse */
	s = &src->outgoing[src->out_count];

	s->dst = dst;
	s->weight = 0;
	s->last_activated = 0;
	s->delta = 0;
	s->flags = 0;
	s->prediction_count = 0;
	s->correct_predictions = 0;
	s->false_predictions = 0;

	src->out_count++;

	/* Add to dst's incoming list */
	spin_lock_irqsave(&dst->lock, flags);

	if (dst->in_count < dst->in_capacity) {
		dst->incoming[dst->in_count++] = src;
	}

	spin_unlock_irqrestore(&dst->lock, flags);

	SMN_DBG("Created synapse %p->%p (weight=%u)\n", src, dst, s->weight);

	return s;
}

/*
 * Find existing synapse from src to dst
 */
struct synapse *smn_find_synapse(struct synaptic_neuron *src,
				 struct synaptic_neuron *dst)
{
	int i;

	if (!src || !dst)
		return NULL;

	for (i = 0; i < src->out_count; i++) {
		if (src->outgoing[i].dst == dst)
			return &src->outgoing[i];
	}

	return NULL;
}

/*
 * Destroy a synapse
 */
void smn_synapse_destroy(struct synaptic_neuron *src, struct synapse *s)
{
	struct synaptic_neuron *dst;
	int i, j;
	unsigned long flags;

	if (!src || !s)
		return;

	dst = s->dst;

	/* Remove from src's outgoing list */
	for (i = 0; i < src->out_count; i++) {
		if (&src->outgoing[i] == s) {
			/* Shift remaining synapses */
			for (j = i; j < src->out_count - 1; j++) {
				src->outgoing[j] = src->outgoing[j + 1];
			}
			src->out_count--;
			break;
		}
	}

	/* Remove from dst's incoming list */
	if (dst) {
		spin_lock_irqsave(&dst->lock, flags);

		for (i = 0; i < dst->in_count; i++) {
			if (dst->incoming[i] == src) {
				/* Shift remaining */
				for (j = i; j < dst->in_count - 1; j++) {
					dst->incoming[j] = dst->incoming[j + 1];
				}
				dst->in_count--;
				break;
			}
		}

		spin_unlock_irqrestore(&dst->lock, flags);
	}
}

/*
 * Strengthen a synapse (learning)
 */
void smn_synapse_strengthen(struct synaptic_neuron *src,
			    struct synaptic_neuron *dst)
{
	struct synapse *s;
	u16 delta;
	unsigned long flags;

	s = smn_find_synapse(src, dst);
	if (!s) {
		s = smn_synapse_create(src, dst);
		if (IS_ERR(s))
			return;
	}

	spin_lock_irqsave(&src->lock, flags);

	/* Calculate weight increase based on current weight */
	delta = smn_global.config.learning_rate *
		(SMN_WEIGHT_MAX - s->weight) / SMN_WEIGHT_MAX;

	s->weight = min(SMN_WEIGHT_MAX, s->weight + delta);
	s->last_activated = smn_time_ms();
	s->flags |= SYNAPSE_FLAG_LEARNING;

	/* Set predictive flag if strong enough */
	if (s->weight > smn_global.config.predictive_threshold) {
		s->flags |= SYNAPSE_FLAG_PREDICTIVE;
	}

	/* Set co-locate flag if very strong */
	if (s->weight > smn_global.config.colocate_threshold) {
		s->flags |= SYNAPSE_FLAG_COLOCATE;
	}

	s->flags &= ~SYNAPSE_FLAG_LEARNING;

	spin_unlock_irqrestore(&src->lock, flags);

	SMN_DBG("Strengthened synapse %p->%p (weight=%u)\n",
		src, dst, s->weight);
}

/*
 * Weaken a synapse (for pruning)
 */
void smn_synapse_weaken(struct synaptic_neuron *src,
		       struct synaptic_neuron *dst)
{
	struct synapse *s;
	unsigned long flags;

	s = smn_find_synapse(src, dst);
	if (!s)
		return;

	spin_lock_irqsave(&src->lock, flags);

	/* Decrease weight */
	if (s->weight > 0)
		s->weight--;

	/* Clear flags if too weak */
	if (s->weight < smn_global.config.predictive_threshold) {
		s->flags &= ~SYNAPSE_FLAG_PREDICTIVE;
	}

	if (s->weight < smn_global.config.colocate_threshold) {
		s->flags &= ~SYNAPSE_FLAG_COLOCATE;
	}

	spin_unlock_irqrestore(&src->lock, flags);
}

/*
 * Dump synapses for a neuron
 */
void smn_dump_synapses(struct synaptic_neuron *neuron)
{
	int i;

	pr_info("SMN Synapses for neuron %p:\n", neuron);
	pr_info("  Outgoing (%u):\n", neuron->out_count);

	for (i = 0; i < neuron->out_count; i++) {
		struct synapse *s = &neuron->outgoing[i];

		pr_info("    -> %p: weight=%u, flags=0x%x\n",
		       s->dst, s->weight, s->flags);
		if (s->flags & SYNAPSE_FLAG_PREDICTIVE)
			pr_cont(" [PREDICTIVE]");
		if (s->flags & SYNAPSE_FLAG_COLOCATE)
			pr_cont(" [COLOCATE]");
		if (s->flags & SYNAPSE_FLAG_BIDIRECTIONAL)
			pr_cont(" [BIDIR]");
		pr_cont("\n");
	}

	pr_info("  Incoming (%u):\n", neuron->in_count);
	for (i = 0; i < neuron->in_count; i++) {
		struct synapse *s = smn_find_synapse(neuron->incoming[i], neuron);

		if (s) {
			pr_info("    <- %p: weight=%u\n",
			       neuron->incoming[i], s->weight);
		}
	}
}
