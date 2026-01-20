# SPDX-License-Identifier: GPL-2.0
#
# Makefile for Synaptic Memory Network (SMN)
#

obj-$(CONFIG_SMN) := smn.o

smn-y := core.o neuron.o synapse.o layer.o learning.o integration.o

ccflags-$(CONFIG_SMN_DEBUG) += -DSMN_DEBUG=1
