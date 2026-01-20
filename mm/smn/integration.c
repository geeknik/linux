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
		struct synapse *s = smn_find_synapse(neuron->incoming[i], neuron);
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

/*
 * Initialize SMN for a specific mm_struct
 * Called when a new address space is created
 */
int smn_init_mm(struct mm_struct *mm)
{
	if (!smn_global.initialized)
		return 0;

	/* TODO: Create per-mm SMN state */
	return 0;
}

/*
 * Cleanup SMN for a specific mm_struct
 * Called when an address space is destroyed
 */
void smn_cleanup_mm(struct mm_struct *mm)
{
	if (!smn_global.initialized)
		return;

	/* TODO: Clean up per-mm SMN state */
}

/*
 * VMA operations integration
 */

/*
 * Called when a VMA is created or modified
 */
void smn_vma_changed(struct vm_area_struct *vma)
{
	if (!smn_global.initialized || !vma)
		return;

	/* TODO: Create VMA-level neuron for coarse-grained tracking */
}

/*
 * Called when a VMA is unmapped
 */
void smn_vma_unmapped(struct vm_area_struct *vma, unsigned long start,
		     unsigned long end)
{
	if (!smn_global.initialized || !vma)
		return;

	/* TODO: Clean up neurons for unmapped pages */
}
