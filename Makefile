SHELL=/bin/sh

BUILD_TYPE=release

OUT_BIN_DIR=bin

BIN_OPENMP=sparse-mat-product-openmp
BIN_CUDA=sparse-mat-product-cuda

CC=$(shell which gcc)
NVCC=$(shell which nvcc) -ccbin='$(CC)'

INCLUDES=include/
CFLAGS=-Wall -W -Wextra -Wshadow -I$(INCLUDES)
CC_CFLAGS=-std=c11
NVCC_CFLAGS=
NVCC_CC_CFLAGS=

COMMON_SRC_FILES := $(shell find ./src/ -type f -name '*.c' -not -path "./src/main*")
COMMON_OBJ_FILES := $(patsubst %.c, %.o, $(COMMON_SRC_FILES))

OPENMP_SPECIFIC_SRC_FILES := $(shell find ./src/main/openmp -type f -name '*.c')
OPENMP_SPECIFIC_OBJ_FILES := $(patsubst %.c, %.o, $(OPENMP_SPECIFIC_SRC_FILES))

CUDA_C_SPECIFIC_SRC_FILES := $(shell find ./src/main/cuda -type f -name \*.c)
CUDA_CU_SPECIFIC_SRC_FILES := $(shell find ./src/main/cuda -type f -name \*.cu)
CUDA_C_SPECIFIC_OBJ_FILES := $(patsubst %.c, %.o, $(CUDA_C_SPECIFIC_SRC_FILES))
CUDA_CU_SPECIFIC_OBJ_FILES := $(patsubst %.cu, %.o, $(CUDA_CU_SPECIFIC_SRC_FILES))

all: openmp cuda

common: $(COMMON_OBJ_FILES)

openmp: common $(OPENMP_SPECIFIC_OBJ_FILES)
	@mkdir -p $(OUT_BIN_DIR)/$(BUILD_TYPE)
	@echo "\tLINK openmp"
	@$(CC) $(COMMON_OBJ_FILES) $(OPENMP_SPECIFIC_OBJ_FILES) -o $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_OPENMP)
	@echo "\tELF $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_OPENMP)"

cuda: common $(CUDA_C_SPECIFIC_OBJ_FILES) $(CUDA_CU_SPECIFIC_OBJ_FILES)
	@mkdir -p $(OUT_BIN_DIR)/$(BUILD_TYPE)
	@echo "\tLINK cuda"
	@$(NVCC) $(COMMON_OBJ_FILES) $(CUDA_C_SPECIFIC_OBJ_FILES) $(CUDA_CU_SPECIFIC_OBJ_FILES) -o $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_CUDA)
	@echo "\tELF $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_CUDA)"

$(CUDA_C_SPECIFIC_OBJ_FILES) $(COMMON_OBJ_FILES): %.o: %.c
	@echo -e '\tCC $<'
	@$(CC) $(CC_CFLAGS) $(CFLAGS) $< -c -o $@

$(OPENMP_SPECIFIC_OBJ_FILES): %.o: %.c
	@echo -e '\tCC $<'
	@$(CC) -fopenmp $(CC_CFLAGS) $(CFLAGS) $< -c -o $@
 
$(CUDA_CU_SPECIFIC_OBJ_FILES): %.o: %.cu
	@echo -e '\tCC $<'
	@$(NVCC) $(NVCC_CFLAGS) -Xcompiler="$(CFLAGS) $(NVCC_CC_CFLAGS)" $< -c -o $@