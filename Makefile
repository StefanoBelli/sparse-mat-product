SHELL=/bin/sh

valid_build_types = release both-debug host-debug device-debug host-debug-device-profile device-profile
BUILD_TYPE ?= release

ifneq ($(filter $(BUILD_TYPE), $(valid_build_types)),)
    $(info BUILD_TYPE=$(BUILD_TYPE) ok)
else
    $(error BUILD_TYPE=$(BUILD_TYPE) is invalid)
endif

OUT_BIN_DIR=bin

BIN_OPENMP=sparse-mat-product-openmp
BIN_CUDA=sparse-mat-product-cuda

CC=$(shell which gcc)
NVCC=$(shell which nvcc) -ccbin='$(CC)'

INCLUDES=include/
CFLAGS=-Wall -W -Wextra -Wshadow -march=native -I$(INCLUDES)
CC_CFLAGS=-std=c11
NVCC_CFLAGS=-Xptxas="--verbose --warn-on-double-precision-use --warn-on-local-memory-usage --warn-on-spills"
NVCC_CC_CFLAGS=
LINK_LIBS=-lcurl

ifeq ($(BUILD_TYPE), release)
	CFLAGS += -O3
else ifeq ($(BUILD_TYPE), both-debug)
	CFLAGS += -O0 -g -ggdb
	NVCC_CFLAGS += --device-debug --source-in-ptx
else ifeq ($(BUILD_TYPE), host-debug)
	CFLAGS += -O0 -g -ggdb
else ifeq ($(BUILD_TYPE), device-debug)
	CFLAGS += -O3
	NVCC_CFLAGS += --device-debug --source-in-ptx
else ifeq ($(BUILD_TYPE), host-debug-device-profile)
	CFLAGS += -O0 -g -ggdb
	NVCC_CFLAGS += --generate-line-info --source-in-ptx
else ifeq ($(BUILD_TYPE), device-profile)
	CFLAGS += -O3
	NVCC_CFLAGS += --generate-line-info --source-in-ptx
endif

COMMON_SRC_FILES := $(shell find ./src/ -type f -name '*.c' -not -path "./src/main*")
COMMON_OBJ_FILES := $(patsubst %.c, %.o, $(COMMON_SRC_FILES))

OPENMP_SPECIFIC_SRC_FILES := $(shell find ./src/main/openmp -type f -name '*.c')
OPENMP_SPECIFIC_OBJ_FILES := $(patsubst %.c, %.o, $(OPENMP_SPECIFIC_SRC_FILES))

CUDA_C_SPECIFIC_SRC_FILES := $(shell find ./src/main/cuda -type f -name \*.c)
CUDA_CU_SPECIFIC_SRC_FILES := $(shell find ./src/main/cuda -type f -name \*.cu)
CUDA_C_SPECIFIC_OBJ_FILES := $(patsubst %.c, %.o, $(CUDA_C_SPECIFIC_SRC_FILES))
CUDA_CU_SPECIFIC_OBJ_FILES := $(patsubst %.cu, %.o, $(CUDA_CU_SPECIFIC_SRC_FILES))

ELF_OUT_OPENMP := $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_OPENMP)
ELF_OUT_CUDA := $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_CUDA) 

$(info compilation flags: CFLAGS=$(CFLAGS))
$(info nvcc flags: NVCC_CFLAGS=$(NVCC_CFLAGS))
$(info libraries to link against: LINK_LIBS=$(LINK_LIBS))
$(info extras: CC_CFLAGS=$(CC_CFLAGS))
$(info extra cxx-host-forwarded nvcc flags: NVCC_CC_CFLAGS=$(NVCC_CC_CFLAGS))

all: openmp cuda

clean: clean-common clean-cuda clean-openmp

common: $(COMMON_OBJ_FILES)

openmp: common $(OPENMP_SPECIFIC_OBJ_FILES)
	@mkdir -p $(OUT_BIN_DIR)/$(BUILD_TYPE)
	@echo "\tOPENMP-LINK"
	@$(CC) \
		$(COMMON_OBJ_FILES) \
		$(OPENMP_SPECIFIC_OBJ_FILES) \
		$(LINK_LIBS) \
		-o $(ELF_OUT_OPENMP)
	@echo "\tELF $(ELF_OUT_OPENMP)"

cuda: common $(CUDA_C_SPECIFIC_OBJ_FILES) $(CUDA_CU_SPECIFIC_OBJ_FILES)
	@mkdir -p $(OUT_BIN_DIR)/$(BUILD_TYPE)
	@echo "\tCUDA-LINK"
	@$(NVCC) \
		$(COMMON_OBJ_FILES) \
		$(CUDA_C_SPECIFIC_OBJ_FILES) \
		$(CUDA_CU_SPECIFIC_OBJ_FILES) \
		$(LINK_LIBS) \
		-o $(ELF_OUT_CUDA)
	@echo "\tELF $(ELF_OUT_CUDA)"

CLEAN_COMMON_OBJS := $(shell find ./src -type f -name '*.o' -not -path "./src/main*")
clean-common:
	@for obj in $(CLEAN_COMMON_OBJS); do echo "\tCLEAN $$obj"; done
	@rm $(CLEAN_COMMON_OBJS)

CLEAN_CUDA_OBJS := $(shell find ./src/main/cuda -type f -name '*.o')
clean-cuda:
	@for obj in $(CLEAN_CUDA_OBJS); do echo "\tCLEAN $$obj"; done
	@rm $(CLEAN_CUDA_OBJS)

CLEAN_OPENMP_OBJS := $(shell find ./src/main/openmp -type f -name '*.o')
clean-openmp:
	@for obj in $(CLEAN_OPENMP_OBJS); do echo "\tCLEAN $$obj"; done
	@rm $(CLEAN_OPENMP_OBJS)

$(CUDA_C_SPECIFIC_OBJ_FILES) $(COMMON_OBJ_FILES): %.o: %.c
	@echo "\tCC $<"
	@$(CC) $(CC_CFLAGS) $(CFLAGS) $< -c -o $@

$(OPENMP_SPECIFIC_OBJ_FILES): %.o: %.c
	@echo "\tCC $<"
	@$(CC) -fopenmp $(CC_CFLAGS) $(CFLAGS) $< -c -o $@
 
$(CUDA_CU_SPECIFIC_OBJ_FILES): %.o: %.cu
	@echo "\tCC $<"
	@$(NVCC) $(NVCC_CFLAGS) -Xcompiler="$(CFLAGS) $(NVCC_CC_CFLAGS)" $< -c -o $@

help:
	@echo ----help----
	@echo Valid targets: all clean common openmp cuda clean-common clean-cuda clean-openmp help
	@echo Valid BUILD_TYPEs: $(valid_build_types)