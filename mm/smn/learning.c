// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Synaptic Memory Network (SMN) - Learning Algorithms
 *
 * Copyright (C) 2026
 *
 * Hebbian learning and Spike-Timing Dependent Plasticity (STDP).
 */

#define pr_fmt(fmt) "smn: learn: " fmt

#include <linux/slab.h>
#include <linux/gfp.h>
#include <linux/kernel.h>
#include <linux/ktime.h>

#include "synaptic.h"

/*
 * Hebbian learning: strengthen co-activated synapses
 * "Neurons that fire together, wire together"
 */
void smn_hebbian_learn(struct synaptic_neuron *pre,
		       struct synaptic_neuron *post,
		       u64 time_delta_ns)
{
	struct synapse *s;
	u16 delta;
	unsigned long flags;

	if (!pre || !post)
		return;

	/* Check co-activation window */
	if (time_delta_ns > smn_global.config.coactivation_window * NSEC_PER_MSEC)
		return;

	/* Find or create synapse */
	s = smn_find_synapse(pre, post);
	if (!s) {
		s = smn_synapse_create(pre, post);
		if (IS_ERR(s))
			return;
	}

	spin_lock_irqsave(&pre->lock, flags);

	/* Weight change depends on:
	 * 1. Both neurons activated recently
	 * 2. Time difference between activations
	 * 3. Current weight (cap at maximum)
	 */

	/* Recent co-activations count more */
	if (time_delta_ns < SMN_RAPID_WINDOW) {
		delta = smn_global.config.learning_rate * 2;
	} else {
		delta = smn_global.config.learning_rate;
	}

	/* Scale by remaining capacity to reach max */
	delta = delta * (SMN_WEIGHT_MAX - s->weight) / SMN_WEIGHT_MAX;

	/* Apply weight change */
	s->weight = min(SMN_WEIGHT_MAX, s->weight + delta);
	s->last_activated = smn_time_ms();

	/* Set predictive flag if strong enough */
	if (s->weight > smn_global.config.predictive_threshold) {
		s->flags |= SYNAPSE_FLAG_PREDICTIVE;
	}

	spin_unlock_irqrestore(&pre->lock, flags);

	SMN_DBG("Hebbian: %p->%p (delta=%u, weight=%u)\n",
		pre, post, delta, s->weight);
}

/*
 * Spike-Timing Dependent Plasticity (STDP)
 * Order matters: causal vs anti-causal
 */
void smn_stdp_learn(struct synaptic_neuron *pre,
		    struct synaptic_neuron *post,
		    u64 pre_time, u64 post_time)
{
	struct synapse *s;
	s64 time_diff;
	u16 delta;
	unsigned long flags;

	if (!pre || !post)
		return;

	s = smn_find_synapse(pre, post);
	if (!s) {
		s = smn_synapse_create(pre, post);
		if (IS_ERR(s))
			return;
	}

	time_diff = post_time - pre_time;

	spin_lock_irqsave(&pre->lock, flags);

	if (time_diff > 0 && time_diff < SMN_STDP_WINDOW) {
		/* Pre before Post: causal, strengthen */
		/* Weight increase decays with time difference */
		delta = SMN_STDP_RATE * (SMN_STDP_WINDOW - time_diff) / SMN_STDP_WINDOW;
		s->weight = min(SMN_WEIGHT_MAX, s->weight + delta);
		s->last_activated = smn_time_ms();

	} else if (time_diff < 0 && -time_diff < SMN_STDP_WINDOW) {
		/* Post before Pre: anti-causal, weaken */
		delta = SMN_STDP_RATE * (SMN_STDP_WINDOW + time_diff) / SMN_STDP_WINDOW;
		s->weight = max(0, s->weight - delta);
	}

	/* Update flags based on new weight */
	if (s->weight > smn_global.config.predictive_threshold) {
		s->flags |= SYNAPSE_FLAG_PREDICTIVE;
	} else {
		s->flags &= ~SYNAPSE_FLAG_PREDICTIVE;
	}

	spin_unlock_irqrestore(&pre->lock, flags);

	SMN_DBG("STDP: %p->%p (diff=%lld, weight=%u)\n",
		pre, post, time_diff, s->weight);
}

/*
 * Learn from recent activations (called after neuron activation)
 * This is the main learning entry point
 */
void smn_learn_from_recent(struct synaptic_neuron *neuron)
{
	struct synaptic_layer *layer;
	struct synaptic_neuron *other;
	int count = 0;

	if (!neuron || !smn_global.initialized)
		return;

	layer = smn_global.layers[LAYER_PAGE];
	if (!layer)
		return;

	/* Look for recently activated neurons in the same layer */
	list_for_each_entry(other, &layer->neuron_list, list) {
		if (other == neuron)
			continue;

		/* Check co-activation timing */
		if (other->last_activation > neuron->last_activation) {
			/* Other activated after us */
			u64 delta = (other->last_activation - neuron->last_activation) * NSEC_PER_MSEC;

			/* Apply learning based on configured mode */
			switch (smn_global.config.learning_mode) {
			case SMN_LEARNING_HEBBIAN:
				smn_hebbian_learn(neuron, other, delta);
				break;
			case SMN_LEARNING_STDP:
				smn_stdp_learn(neuron, other,
					       neuron->last_activation * NSEC_PER_MSEC,
					       other->last_activation * NSEC_PER_MSEC);
				break;
			case SMN_LEARNING_HYBRID:
				smn_hebbian_learn(neuron, other, delta);
				smn_stdp_learn(neuron, other,
					      neuron->last_activation * NSEC_PER_MSEC,
					      other->last_activation * NSEC_PER_MSEC);
				break;
			default:
				break;
			}

		} else {
			/* We activated after other */
			u64 delta = (neuron->last_activation - other->last_activation) * NSEC_PER_MSEC;

			switch (smn_global.config.learning_mode) {
			case SMN_LEARNING_HEBBIAN:
				smn_hebbian_learn(other, neuron, delta);
				break;
			case SMN_LEARNING_STDP:
				smn_stdp_learn(other, neuron,
					       other->last_activation * NSEC_PER_MSEC,
					       neuron->last_activation * NSEC_PER_MSEC);
				break;
			case SMN_LEARNING_HYBRID:
				smn_hebbian_learn(other, neuron, delta);
				smn_stdp_learn(other, neuron,
					      other->last_activation * NSEC_PER_MSEC,
					      neuron->last_activation * NSEC_PER_MSEC);
				break;
			default:
				break;
			}
		}

		/* Limit scan for performance */
		if (++count > 100)
			break;
	}
}

/*
 * Prefetch from neuron along its synapses
 */
void smn_prefetch_from_neuron(struct synaptic_neuron *neuron)
{
	struct synapse *s;
	int i;

	if (!neuron || !smn_global.config.enable_prefetch)
		return;

	for (i = 0; i < neuron->out_count; i++) {
		s = &neuron->outgoing[i];

		/* Only prefetch strong, predictive synapses */
		if (!(s->flags & SYNAPSE_FLAG_PREDICTIVE))
			continue;
		if (s->weight < smn_global.config.predictive_threshold)
			continue;

		/* Mark prediction */
		s->prediction_count++;

		/* TODO: Trigger async readahead */
		/* For now, just mark the destination as predicted */
		if (s->dst) {
			s->dst->flags |= NEURON_FLAG_PREDICTED;
		}
	}
}

/*
 * Prefetch along synapses (called during activation)
 */
void smn_prefetch_along_synapses(struct synaptic_neuron *neuron)
{
	smn_prefetch_from_neuron(neuron);
}

/*
 * Optimize NUMA placement based on synaptic connections
 */
void smn_optimize_placement(struct synaptic_neuron *neuron)
{
	struct synapse *s;
	struct synapse *strongest = NULL;
	u16 max_weight = 0;
	int i;

	if (!neuron || !smn_global.config.enable_colocate)
		return;

	/* Find strongest bidirectional synapse */
	for (i = 0; i < neuron->out_count; i++) {
		s = &neuron->outgoing[i];

		if (s->flags & SYNAPSE_FLAG_BIDIRECTIONAL &&
		    s->weight > max_weight) {
			strongest = s;
			max_weight = s->weight;
		}
	}

	if (strongest && max_weight > smn_global.config.colocate_threshold) {
		/* Migrate to be with strongest neighbor */
		int target_nid = strongest->dst->nid;
		int current_nid = neuron->nid;

		if (target_nid >= 0 && target_nid != current_nid) {
			/* TODO: Trigger NUMA migration */
			SMN_DBG("Would migrate neuron %p to node %d\n",
				neuron, target_nid);
		}
	}
}

/*
 * Feedback for predictions (called when predicted page is accessed)
 */
void smn_prediction_feedback(struct synaptic_neuron *predicted)
{
	struct synaptic_neuron **incoming;
	int i;
	unsigned long flags;

	if (!predicted)
		return;

	incoming = predicted->incoming;

	for (i = 0; i < predicted->in_count; i++) {
		struct synaptic_neuron *src = incoming[i];
		struct synapse *s;

		if (!src)
			continue;

		s = smn_find_synapse(src, predicted);
		if (s) {
			s->correct_predictions++;

			/* Strengthen accurate predictors */
			spin_lock_irqsave(&src->lock, flags);
			u16 bonus = min(100, 1000 / (s->correct_predictions + 1));
			s->weight = min(SMN_WEIGHT_MAX, s->weight + bonus);
			spin_unlock_irqrestore(&src->lock, flags);
		}
	}
}
