// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Synaptic Memory Network (SMN) - Integration Layer
 *
 * Copyright (C) 2026
 *
 * Integration between SMN and the main MM subsystem.
 * Provides hooks for page faults, memory access, and reclaim.
 */

#define pr_fmt(fmt) "smn: " fmt

#include <linux/mm.h>
#include <linux/pagemap.h>
#include <linux/rmap.h>
#include <linux/swap.h>
#include <linux/mm_inline.h>
#include <linux/slab.h>
#include <linux/hashtable.h>

#include "synaptic.h"

/*
 * Handle page fault with SMN integration
 * Called from the main page fault handler after page is allocated
 */
vm_fault_t smn_handle_page_fault(struct vm_fault *vmf)
{
	struct page *page;
	struct synaptic_neuron *neuron;

	if (!smn_global.initialized)
		return VM_FAULT_NOPAGE;

	if (!vmf || !vmf->page)
		return VM_FAULT_NOPAGE;

	page = vmf->page;

	/* Mark page as accessed (triggers SMN learning) */
	smn_mark_page_accessed(page);

	/* Find the neuron for this page */
	neuron = smn_page_to_neuron(page);
	if (neuron) {
		/* Optimize NUMA placement based on synaptic connections */
		smn_optimize_placement(neuron);

		/* Provide feedback if this was a predicted access */
		if (neuron->flags & NEURON_FLAG_PREDICTED) {
			smn_prediction_feedback(neuron);
			neuron->flags &= ~NEURON_FLAG_PREDICTED;
		}
	}

	return VM_FAULT_NOPAGE;
}

/*
 * Hook for page access notification
 * This can be called from various MM paths to notify SMN of page access
 */
void smn_notify_page_access(struct page *page)
{
	if (!smn_global.initialized || !page)
		return;

	smn_mark_page_accessed(page);
}

/*
 * Check if page should be reclaimed (SMN-aware reclaim)
 * Returns true if page can be reclaimed, false if protected
 */
bool smn_page_can_reclaim(struct page *page)
{
	if (!smn_global.initialized || !page)
		return true;

	return smn_page_should_reclaim(page);
}

/*
 * Estimate synaptic importance for a page
 * Returns 0-1000 score (higher = more important, don't reclaim)
 */
u32 smn_page_importance(struct page *page)
{
	struct synaptic_neuron *neuron;
	u32 importance = 0;
	int i;

	if (!smn_global.initialized || !page)
		return 0;

	neuron = smn_page_to_neuron(page);
	if (!neuron)
		return 0;

	/* Base importance on activation rate */
	importance = min(500, neuron->activation_rate * 10);

	/* Add synaptic strength */
	for (i = 0; i < neuron->out_count; i++) {
		importance += neuron->outgoing[i].weight;
	}

	/* Consider incoming synapses */
	for (i = 0; i < neuron->in_count; i++) {
		struct synapse *s =
			smn_find_synapse(neuron->incoming[i], neuron);
		if (s)
			importance += s->weight / 2;
	}

	return min(1000, importance);
}

/*
 * SMN reclaim iterator hook
 * Called by reclaim to check if page should be skipped
 */
bool smn_reclaim_skip_page(struct page *page)
{
	if (!smn_global.initialized || !page)
		return false;

	/* Pages with high synaptic importance are protected */
	return smn_page_importance(page) > 700;
}

static struct smn_mm *smn_mm_alloc(struct mm_struct *mm)
{
	struct smn_mm *smn_mm;

	smn_mm = kzalloc(sizeof(*smn_mm), GFP_KERNEL);
	if (!smn_mm)
		return NULL;

	smn_mm->mm = mm;
	spin_lock_init(&smn_mm->lock);

	return smn_mm;
}

static void smn_mm_free(struct smn_mm *smn_mm)
{
	int i;

	if (!smn_mm)
		return;

	for (i = 0; i < LAYER_TYPE_MAX; i++) {
		if (smn_mm->layers[i])
			smn_layer_destroy(smn_mm->layers[i]);
	}

	kfree(smn_mm);
}

struct smn_mm *smn_get_mm(struct mm_struct *mm)
{
	struct smn_mm *smn_mm;
	unsigned long flags;

	if (!smn_global.initialized || !mm)
		return NULL;

	spin_lock_irqsave(&smn_global.mm_hash_lock, flags);
	hash_for_each_possible(smn_global.mm_hash, smn_mm, node,
			       (unsigned long)mm) {
		if (smn_mm->mm == mm) {
			spin_unlock_irqrestore(&smn_global.mm_hash_lock, flags);
			return smn_mm;
		}
	}
	spin_unlock_irqrestore(&smn_global.mm_hash_lock, flags);

	return NULL;
}

int smn_init_mm(struct mm_struct *mm)
{
	struct smn_mm *smn_mm;
	unsigned long flags;

	if (!smn_global.initialized || !mm)
		return 0;

	if (smn_get_mm(mm))
		return 0;

	smn_mm = smn_mm_alloc(mm);
	if (!smn_mm)
		return -ENOMEM;

	spin_lock_irqsave(&smn_global.mm_hash_lock, flags);
	hash_add(smn_global.mm_hash, &smn_mm->node, (unsigned long)mm);
	spin_unlock_irqrestore(&smn_global.mm_hash_lock, flags);

	return 0;
}

void smn_cleanup_mm(struct mm_struct *mm)
{
	struct smn_mm *smn_mm;
	unsigned long flags;

	if (!smn_global.initialized || !mm)
		return;

	spin_lock_irqsave(&smn_global.mm_hash_lock, flags);
	hash_for_each_possible(smn_global.mm_hash, smn_mm, node,
			       (unsigned long)mm) {
		if (smn_mm->mm == mm) {
			hash_del(&smn_mm->node);
			spin_unlock_irqrestore(&smn_global.mm_hash_lock, flags);
			smn_mm_free(smn_mm);
			return;
		}
	}
	spin_unlock_irqrestore(&smn_global.mm_hash_lock, flags);
}

static struct synaptic_neuron *smn_vma_get_or_create(struct vm_area_struct *vma)
{
	struct synaptic_neuron *neuron;
	struct synaptic_layer *layer;

	if (!smn_global.initialized || !vma)
		return NULL;

	layer = smn_global.layers[LAYER_VMA];
	if (!layer)
		return NULL;

	neuron = smn_layer_find_neuron(layer, (unsigned long)vma);
	if (neuron)
		return neuron;

	neuron = smn_neuron_create(NEURON_VMA, vma);
	if (!neuron)
		return NULL;

	neuron->id.vma = vma;
	if (smn_layer_add_neuron(layer, neuron)) {
		smn_neuron_destroy(neuron);
		return NULL;
	}

	return neuron;
}

void smn_vma_changed(struct vm_area_struct *vma)
{
	struct synaptic_neuron *neuron;

	if (!smn_global.initialized || !vma)
		return;

	neuron = smn_vma_get_or_create(vma);
	if (neuron)
		smn_neuron_activate(neuron);
}

void smn_vma_unmapped(struct vm_area_struct *vma, unsigned long start,
		      unsigned long end)
{
	struct synaptic_neuron *neuron;
	struct synaptic_layer *layer;
	int i;

	if (!smn_global.initialized || !vma)
		return;

	layer = smn_global.layers[LAYER_VMA];
	if (!layer)
		return;

	neuron = smn_layer_find_neuron(layer, (unsigned long)vma);
	if (!neuron)
		return;

	for (i = 0; i < neuron->in_count; i++) {
		struct synaptic_neuron *src = neuron->incoming[i];
		if (src) {
			struct synapse *s = smn_find_synapse(src, neuron);
			if (s)
				smn_synapse_destroy(src, s);
		}
	}

	smn_layer_remove_neuron(layer, neuron);
	smn_neuron_destroy(neuron);
}

void smn_page_free(struct page *page)
{
	struct synaptic_neuron *neuron;
	struct synaptic_layer *layer;
	int i;

	if (!smn_global.initialized || !page)
		return;

	layer = smn_global.layers[LAYER_PAGE];
	if (!layer)
		return;

	neuron = smn_layer_find_neuron(layer, page_to_pfn(page));
	if (!neuron)
		return;

	for (i = 0; i < neuron->in_count; i++) {
		struct synaptic_neuron *src = neuron->incoming[i];
		if (src) {
			struct synapse *s = smn_find_synapse(src, neuron);
			if (s)
				smn_synapse_destroy(src, s);
		}
	}

	smn_layer_remove_neuron(layer, neuron);
	smn_neuron_destroy(neuron);
}
