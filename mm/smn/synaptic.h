/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Synaptic Memory Network (SMN)
 *
 * Copyright (C) 2026
 *
 * Bio-inspired memory management using neural network principles.
 * Memory regions are connected by synapses that strengthen and weaken
 * based on access patterns, enabling the kernel to learn and optimize.
 */

#ifndef _MM_SMN_SYNAPTIC_H
#define _MM_SMN_SYNAPTIC_H

#include <linux/types.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>
#include <linux/radix-tree.h>
#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/workqueue.h>
#include <linux/mm.h>
#include <linux/page-flags.h>
#include <linux/hashtable.h>

/*
 * Configuration Parameters
 */
#define SMN_NAME "smn"

/* Synaptic weights */
#define SMN_WEIGHT_MAX 1000 /* Maximum synapse weight */
#define SMN_WEIGHT_MIN 0 /* Minimum synapse weight */
#define SMN_PREDICTIVE_THRESHOLD 500 /* Weight to enable prefetch */
#define SMN_COLOCATE_THRESHOLD 700 /* Weight to enable co-location */
#define SMN_RECLAIM_PROTECT_THRESHOLD 400 /* Weight to protect from reclaim */
#define SMN_MIN_WEIGHT 50 /* Minimum weight before pruning */

/* Timing parameters (in nanoseconds) */
#define SMN_COACTIVATION_WINDOW (100 * NSEC_PER_MSEC) /* 100ms */
#define SMN_RAPID_WINDOW (10 * NSEC_PER_MSEC) /* 10ms */
#define SMN_STDP_WINDOW (50 * NSEC_PER_MSEC) /* 50ms */

/* Learning rates */
#define SMN_HEBBIAN_LEARNING_RATE 10 /* Weight units per co-activation */
#define SMN_STDP_RATE 20 /* Weight units for STDP */
#define SMN_DECAY_RATE 1000 /* ms per weight unit decay */

/* Capacity limits */
#define SMN_MAX_SYNAPSES_PER_NEURON 16 /* Outgoing synapses per neuron */
#define SMN_NEURON_INITIAL_CAPACITY 256 /* Initial neurons per layer */

/* Reclaim thresholds */
#define SMN_RECLAIM_IDLE_THRESHOLD (30 * 1000) /* 30 seconds in ms */
#define SMN_ACTIVE_RATE_THRESHOLD 10 /* Hz */

/* Debug */
#define SMN_DEBUG 0

/* Sampling: process 1/N accesses. Batching: learn after N activations or Nms */
#define SMN_SAMPLE_RATE 16
#define SMN_BATCH_SIZE 32
#define SMN_BATCH_INTERVAL_MS 10

/*
 * Enums
 */

/**
 * enum neuron_type - Type of neuron (memory region granularity)
 * @NEURON_PAGE: Single page level (4KB)
 * @NEURON_VMA: VMA level (coarser granularity)
 * @NEURON_SUPERPAGE: Huge page level (2MB/1GB)
 * @NEURON_REGION: User-defined memory region
 */
enum neuron_type {
	NEURON_PAGE = 0,
	NEURON_VMA,
	NEURON_SUPERPAGE,
	NEURON_REGION,
	NEURON_TYPE_MAX
};

/**
 * enum layer_type - Type of synaptic layer
 * @LAYER_PAGE: Page-level neurons (fine-grained, most neurons)
 * @LAYER_VMA: VMA-level neurons (medium granularity)
 * @LAYER_REGION: Region-level neurons (coarse)
 * @LAYER_GLOBAL: System-wide correlations
 */
enum layer_type {
	LAYER_PAGE = 0,
	LAYER_VMA,
	LAYER_REGION,
	LAYER_GLOBAL,
	LAYER_TYPE_MAX
};

/**
 * enum smn_learning_mode - Learning algorithm to use
 * @SMN_LEARNING_NONE: No learning (static)
 * @SMN_LEARNING_HEBBIAN: Basic Hebbian learning
 * @SMN_LEARNING_STDP: Spike-Timing Dependent Plasticity
 * @SMN_LEARNING_HYBRID: Combined Hebbian + STDP
 */
enum smn_learning_mode {
	SMN_LEARNING_NONE = 0,
	SMN_LEARNING_HEBBIAN,
	SMN_LEARNING_STDP,
	SMN_LEARNING_HYBRID,
	SMN_LEARNING_MAX
};

/*
 * Core Data Structures
 */

/**
 * struct synapse - Connection between two neurons
 * @dst: Destination neuron
 * @weight: Synapse strength (0-1000)
 * @last_activated: Time (in ms from boot) of last activation
 * @delta: Recent weight change (for learning)
 * @flags: Synapse properties (see SYNAPSE_FLAG_*)
 * @prediction_count: Number of times this synapse correctly predicted access
 * @correct_predictions: Correct prediction counter
 * @false_predictions: False prediction counter
 */
struct synapse {
	struct synaptic_neuron *dst;
	u16 weight;
	u16 last_activated;
	s8 delta;
	u8 flags;

	/* Statistics */
	u32 prediction_count;
	u32 correct_predictions;
	u32 false_predictions;
};

/* Synapse flags */
#define SYNAPSE_FLAG_BIDIRECTIONAL BIT(0) /* Mutual activation */
#define SYNAPSE_FLAG_PREDICTIVE BIT(1) /* Prefetch enabled */
#define SYNAPSE_FLAG_COLOCATE BIT(2) /* Try to place together */
#define SYNAPSE_FLAG_PROTECT BIT(3) /* Protect from reclaim */
#define SYNAPSE_FLAG_LEARNING BIT(4) /* Currently being updated */

/**
 * struct neuron_stats - Statistics for a neuron
 * @total_activations: Total number of activations
 * @correct_predictions: Times this neuron was correctly predicted
 * @false_predictions: Times this neuron was incorrectly predicted
 * @prediction_accuracy: Accuracy percentage (0-100)
 */
struct neuron_stats {
	u64 total_activations;
	u64 correct_predictions;
	u64 false_predictions;
	u16 prediction_accuracy;
};

/**
 * struct synaptic_neuron - Represents a memory region that can activate
 * @id: Identification (union of possible types)
 * @type: Neuron type (page, VMA, etc.)
 * @last_activation: Last time this neuron fired (ms from boot)
 * @prev_activation: Previous activation time (for rate calculation)
 * @activation_count: Total activations
 * @activation_rate: Activations per second (EMA)
 * @outgoing: Outgoing synapses (array)
 * @out_capacity: Allocated capacity for outgoing synapses
 * @out_count: Actual number of outgoing synapses
 * @incoming: Incoming synapses (array of pointers)
 * @in_capacity: Allocated capacity for incoming synapses
 * @in_count: Actual number of incoming synapses
 * @layer: Which layer this neuron belongs to
 * @nid: Preferred NUMA node
 * @flags: Neuron flags
 * @stats: Statistics
 * @lock: Spinlock protecting this neuron
 * @list: List node for layer's neuron list
 */
struct synaptic_neuron {
	union {
		struct page *page;
		struct vm_area_struct *vma;
		phys_addr_t phys_addr;
		void *raw;
	} id;

	enum neuron_type type;

	/* State */
	u64 last_activation;
	u64 prev_activation;
	u64 last_activation_ns;
	u64 prev_activation_ns;
	u32 activation_count;
	u16 activation_rate;

	/* Synapses */
	struct synapse *outgoing;
	u16 out_capacity;
	u16 out_count;

	struct synaptic_neuron **incoming;
	u16 in_capacity;
	u16 in_count;

	/* Position */
	struct synaptic_layer *layer;
	int nid;

	/* Flags and stats */
	u8 flags;
	struct neuron_stats stats;

	/* Synchronization */
	spinlock_t lock;
	struct list_head list;
	struct list_head recent;
};

/* Neuron flags */
#define NEURON_FLAG_ACTIVE BIT(0) /* Recently active */
#define NEURON_FLAG_PREDICTED BIT(1) /* Was predicted access */
#define NEURON_FLAG_PROTECTED BIT(2) /* Protected from reclaim */
#define NEURON_FLAG_MIGRATING BIT(3) /* Being migrated */

/**
 * struct layer_stats - Statistics for a synaptic layer
 * @total_activations: Total activations in this layer
 * @avg_synapse_weight: Average synapse weight
 * @connectivity: Average synapses per neuron
 * @neuron_count: Current number of neurons
 */
struct layer_stats {
	u64 total_activations;
	u16 avg_synapse_weight;
	u16 connectivity;
	u32 neuron_count;
};

/**
 * struct synaptic_layer - Layer of neurons (hierarchical organization)
 * @type: Layer type
 * @neuron_capacity: Maximum neurons this layer can hold
 * @neuron_count: Current number of neurons
 * @neurons: Array of neuron pointers
 * @neuron_tree: Radix tree for fast neuron lookup by address
 * @prev_layer: Previous layer (finer granularity)
 * @next_layer: Next layer (coarser granularity)
 * @inter_synapses: Synapses between layers (optional)
 * @stats: Layer statistics
 * @lock: Mutex protecting layer structure
 * @neuron_list: List of all neurons in this layer
 * @prune_work: Work item for periodic pruning
 */
struct synaptic_layer {
	enum layer_type type;
	u32 neuron_capacity;
	u32 neuron_count;
	struct synaptic_neuron **neurons;
	struct radix_tree_root neuron_tree;

	struct synaptic_layer *prev_layer;
	struct synaptic_layer *next_layer;
	struct synapse *inter_synapses;

	struct layer_stats stats;
	struct mutex lock;
	struct list_head neuron_list;
	struct list_head recent_list;
	spinlock_t recent_lock;
	u32 recent_count;
	struct delayed_work prune_work;

	atomic_t sample_counter;
	struct synaptic_neuron *batch_queue[SMN_BATCH_SIZE];
	atomic_t batch_count;
	u64 batch_start_time;
	struct work_struct batch_work;
};

/**
 * struct smn_config - Configuration parameters for SMN
 * @learning_mode: Learning algorithm to use
 * @learning_rate: How fast synapses strengthen
 * @decay_rate: How fast synapses weaken
 * @coactivation_window: Time window for learning (ms)
 * @predictive_threshold: Weight to enable prefetch
 * @colocate_threshold: Weight to enable co-location
 * @reclaim_protection: Weight to protect from reclaim
 * @max_synapses_per_neuron: Memory limit
 * @prune_interval: Pruning frequency (ms)
 * @enable_prefetch: Enable synaptic prefetching
 * @enable_colocate: Enable synaptic co-location
 */
struct smn_config {
	enum smn_learning_mode learning_mode;
	u16 learning_rate;
	u16 decay_rate;
	u32 coactivation_window;
	u16 predictive_threshold;
	u16 colocate_threshold;
	u16 reclaim_protection;
	u16 max_synapses_per_neuron;
	u32 prune_interval;
	bool enable_prefetch;
	bool enable_colocate;
};

/**
 * struct smn_system - Main SMN subsystem structure
 * @layers: Array of layers (LAYER_PAGE through LAYER_GLOBAL)
 * @config: Current configuration
 * @initialized: Whether SMN is initialized
 * @lock: Global lock
 * @stats_wq: Workqueue for statistics
 * @prune_wq: Workqueue for pruning
 * @mm_hash: Hash table mapping mm_struct to smn_mm
 * @mm_hash_lock: Protects mm_hash
 */
struct smn_system {
	struct synaptic_layer *layers[LAYER_TYPE_MAX];
	struct smn_config config;
	bool initialized;
	struct mutex lock;

	struct workqueue_struct *stats_wq;
	struct workqueue_struct *prune_wq;

	DECLARE_HASHTABLE(mm_hash, 8);
	spinlock_t mm_hash_lock;
};

/**
 * struct smn_mm - Per-process synaptic memory state
 * @mm: The associated mm_struct
 * @layers: Per-process layers (for process-local learning)
 * @node: List node for smn_global.mm_list
 * @lock: Protects this structure
 * @neuron_count: Total neurons in this mm's scope
 * @synapse_count: Total synapses in this mm's scope
 *
 * Each mm_struct gets its own SMN context for security isolation.
 * Neurons in one process cannot form synapses to another process's neurons.
 */
struct smn_mm {
	struct mm_struct *mm;
	struct synaptic_layer *layers[LAYER_TYPE_MAX];
	struct hlist_node node;
	spinlock_t lock;
	u32 neuron_count;
	u32 synapse_count;
};

/*
 * Global SMN system instance
 */
extern struct smn_system smn_global;

/*
 * Core API Functions
 */

/* Initialization */
int smn_init(void);
void smn_exit(void);

/* Layer management */
struct synaptic_layer *smn_layer_create(enum layer_type type);
void smn_layer_destroy(struct synaptic_layer *layer);
int smn_layer_add_neuron(struct synaptic_layer *layer,
			 struct synaptic_neuron *neuron);
struct synaptic_neuron *smn_layer_find_neuron(struct synaptic_layer *layer,
					      unsigned long addr);

/* Neuron management */
struct synaptic_neuron *smn_neuron_create(enum neuron_type type, void *id);
void smn_neuron_destroy(struct synaptic_neuron *neuron);
void smn_neuron_activate(struct synaptic_neuron *neuron);
struct synaptic_neuron *smn_page_to_neuron(struct page *page);
struct page *smn_neuron_to_page(struct synaptic_neuron *neuron);

/* Synapse operations */
struct synapse *smn_synapse_create(struct synaptic_neuron *src,
				   struct synaptic_neuron *dst);
void smn_synapse_destroy(struct synaptic_neuron *src, struct synapse *s);
struct synapse *smn_find_synapse(struct synaptic_neuron *src,
				 struct synaptic_neuron *dst);
void smn_synapse_strengthen(struct synaptic_neuron *src,
			    struct synaptic_neuron *dst);
void smn_synapse_weaken(struct synaptic_neuron *src,
			struct synaptic_neuron *dst);

/* Learning algorithms */
void smn_hebbian_learn(struct synaptic_neuron *pre,
		       struct synaptic_neuron *post, u64 time_delta_ns);
void smn_stdp_learn(struct synaptic_neuron *pre, struct synaptic_neuron *post,
		    u64 pre_time, u64 post_time);
void smn_learn_from_recent(struct synaptic_neuron *neuron);

/* Prefetch and optimization */
void smn_prefetch_from_neuron(struct synaptic_neuron *neuron);
void smn_prefetch_along_synapses(struct synaptic_neuron *neuron);
void smn_optimize_placement(struct synaptic_neuron *neuron);
void smn_prediction_feedback(struct synaptic_neuron *predicted);

/* Reclaim integration */
bool smn_should_reclaim(struct synaptic_neuron *neuron);
bool smn_page_should_reclaim(struct page *page);

/* Maintenance */
void smn_prune_synapses(struct synaptic_layer *layer);
void smn_decay_synapses(struct synaptic_layer *layer);

/* Page fault integration */
vm_fault_t smn_handle_page_fault(struct vm_fault *vmf);
void smn_mark_page_accessed(struct page *page);
void smn_notify_page_access(struct page *page);
bool smn_page_can_reclaim(struct page *page);
u32 smn_page_importance(struct page *page);
bool smn_reclaim_skip_page(struct page *page);

/* MM integration */
int smn_init_mm(struct mm_struct *mm);
void smn_cleanup_mm(struct mm_struct *mm);
struct smn_mm *smn_get_mm(struct mm_struct *mm);
void smn_vma_changed(struct vm_area_struct *vma);
void smn_vma_unmapped(struct vm_area_struct *vma, unsigned long start,
		      unsigned long end);

/* Statistics and debugging */
void smn_stats_print(void);
void smn_dump_neurons(struct synaptic_layer *layer);
void smn_dump_synapses(struct synaptic_neuron *neuron);

/* Layer functions that need to be exported */
void smn_layer_start_pruning(struct synaptic_layer *layer);
void smn_layer_update_stats(struct synaptic_layer *layer);
u64 smn_layer_count_synapses(struct synaptic_layer *layer);
int smn_layer_remove_neuron(struct synaptic_layer *layer,
			    struct synaptic_neuron *neuron);

/* Page lifecycle hooks */
void smn_page_free(struct page *page);

/* Batch processing */
void smn_batch_work_func(struct work_struct *work);

/*
 * Utility Functions
 */

static inline u64 smn_time_ms(void)
{
	return jiffies_to_msecs(jiffies);
}

static inline u64 smn_time_ns(void)
{
	return ktime_get_ns();
}

static inline bool smn_is_active(u64 last_activation, u64 threshold_ms)
{
	return (smn_time_ms() - last_activation) < threshold_ms;
}

/*
 * Debugging macros
 */
#if SMN_DEBUG
#define SMN_DBG(fmt, args...) pr_info("SMN: " fmt, ##args)
#define SMN_DBG_NEURON(n, fmt, args...) \
	pr_info("SMN neuron %p: " fmt, n, ##args)
#else
#define SMN_DBG(fmt, args...) \
	do {                  \
	} while (0)
#define SMN_DBG_NEURON(n, fmt, args...) \
	do {                            \
	} while (0)
#endif

#endif /* _MM_SMN_SYNAPTIC_H */
