SHELL=/bin/sh

valid_build_types = release host-sanitize both-debug host-debug device-debug host-debug-device-profile device-profile
BUILD_TYPE ?= release

ifneq ($(filter $(BUILD_TYPE), $(valid_build_types)),)
    $(info BUILD_TYPE=$(BUILD_TYPE) ok)
else
    $(error BUILD_TYPE=$(BUILD_TYPE) is invalid)
endif

OUT_BIN_DIR=bin

BIN_SERIAL=sparse-mat-product-serial
BIN_OPENMP=sparse-mat-product-openmp
BIN_CUDA=sparse-mat-product-cuda

CC=$(shell which gcc)
NVCC=$(shell which nvcc) -ccbin='$(CC)'

INCLUDES=include/
CFLAGS=-Wall -W -Wextra -Wshadow -march=native -I$(INCLUDES)
CC_CFLAGS=-std=c11
NVCC_CFLAGS=-Xptxas="--verbose --warn-on-double-precision-use --warn-on-local-memory-usage --warn-on-spills"
NVCC_CC_CFLAGS=
LINK_LIBS=
SANITIZERS=

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
else ifeq ($(BUILD_TYPE), host-sanitize)
	SANITIZERS= \
		-fsanitize=address \
		-fsanitize=leak \
		-fsanitize=undefined \
		-fsanitize=pointer-compare \
		-fsanitize=pointer-subtract
	CFLAGS += -g -O0 $(SANITIZERS)
endif

COMMON_SRC_FILES := $(shell find ./src/ -type f -name '*.c' -not -path "./src/main*")
COMMON_OBJ_FILES := $(patsubst %.c, %.o, $(COMMON_SRC_FILES))

SERIAL_SPECIFIC_SRC_FILES := $(shell find ./src/main/serial -type f -name '*.c')
SERIAL_SPECIFIC_OBJ_FILES := $(patsubst %.c, %.o, $(SERIAL_SPECIFIC_SRC_FILES))

OPENMP_SPECIFIC_SRC_FILES := $(shell find ./src/main/openmp -type f -name '*.c')
OPENMP_SPECIFIC_OBJ_FILES := $(patsubst %.c, %.o, $(OPENMP_SPECIFIC_SRC_FILES))

CUDA_C_SPECIFIC_SRC_FILES := $(shell find ./src/main/cuda -type f -name \*.c)
CUDA_CU_SPECIFIC_SRC_FILES := $(shell find ./src/main/cuda -type f -name \*.cu)
CUDA_C_SPECIFIC_OBJ_FILES := $(patsubst %.c, %.o, $(CUDA_C_SPECIFIC_SRC_FILES))
CUDA_CU_SPECIFIC_OBJ_FILES := $(patsubst %.cu, %.o, $(CUDA_CU_SPECIFIC_SRC_FILES))

ELF_OUT_SERIAL := $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_SERIAL)
ELF_OUT_OPENMP := $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_OPENMP)
ELF_OUT_CUDA := $(OUT_BIN_DIR)/$(BUILD_TYPE)/$(BIN_CUDA) 

$(info compilation flags: CFLAGS=$(CFLAGS))
$(info nvcc flags: NVCC_CFLAGS=$(NVCC_CFLAGS))
$(info libraries to link against: LINK_LIBS=$(LINK_LIBS))
$(info extras: CC_CFLAGS=$(CC_CFLAGS))
$(info extra cxx-host-forwarded nvcc flags: NVCC_CC_CFLAGS=$(NVCC_CC_CFLAGS))

all: serial openmp cuda

clean: clean-serial clean-common clean-openmp clean-cuda

common: $(COMMON_OBJ_FILES)

serial: common $(SERIAL_SPECIFIC_OBJ_FILES)
	@mkdir -p $(OUT_BIN_DIR)/$(BUILD_TYPE)
	@printf '\tSERIAL-LINK\n'
	@$(CC) \
		$(COMMON_OBJ_FILES) \
		$(SERIAL_SPECIFIC_OBJ_FILES) \
		$(LINK_LIBS) \
		$(SANITIZERS) \
		-o $(ELF_OUT_SERIAL)
	@printf '\tELF $(ELF_OUT_SERIAL)\n'

openmp: common $(OPENMP_SPECIFIC_OBJ_FILES)
	@mkdir -p $(OUT_BIN_DIR)/$(BUILD_TYPE)
	@printf '\tOPENMP-LINK\n'
	@$(CC) \
		$(COMMON_OBJ_FILES) \
		$(OPENMP_SPECIFIC_OBJ_FILES) \
		$(LINK_LIBS) \
		$(SANITIZERS) \
		-o $(ELF_OUT_OPENMP)
	@printf '\tELF $(ELF_OUT_OPENMP)\n'

cuda: common $(CUDA_C_SPECIFIC_OBJ_FILES) $(CUDA_CU_SPECIFIC_OBJ_FILES)
	@mkdir -p $(OUT_BIN_DIR)/$(BUILD_TYPE)
	@printf '\tCUDA-LINK\n'
	@$(NVCC) \
		$(COMMON_OBJ_FILES) \
		$(CUDA_C_SPECIFIC_OBJ_FILES) \
		$(CUDA_CU_SPECIFIC_OBJ_FILES) \
		$(LINK_LIBS) \
		-Xcompiler="$(SANITIZERS)" \
		-o $(ELF_OUT_CUDA)
	@printf '\tELF $(ELF_OUT_CUDA)\n'

CLEAN_COMMON_OBJS := $(shell find ./src -type f -name '*.o' -not -path "./src/main*")
clean-common:
	@for file in $(CLEAN_COMMON_OBJS); do \
        printf "\tCLEAN $${file}\n"; \
        rm $$file; \
    done	

CLEAN_CUDA_OBJS := $(shell find ./src/main/cuda -type f -name '*.o')
clean-cuda:
	@for file in $(CLEAN_CUDA_OBJS); do \
        printf "\tCLEAN $${file}\n"; \
        rm $$file; \
    done

CLEAN_OPENMP_OBJS := $(shell find ./src/main/openmp -type f -name '*.o')
clean-openmp:
	@for file in $(CLEAN_OPENMP_OBJS); do \
        printf "\tCLEAN $${file}\n"; \
        rm $$file; \
    done

CLEAN_SERIAL_OBJS := $(shell find ./src/main/serial -type f -name '*.o')
clean-serial:
	@for file in $(CLEAN_SERIAL_OBJS); do \
        printf "\tCLEAN $${file}\n"; \
        rm $$file; \
    done

$(CUDA_C_SPECIFIC_OBJ_FILES) $(COMMON_OBJ_FILES) $(SERIAL_SPECIFIC_OBJ_FILES): %.o: %.c
	@printf '\tCC $<\n'
	@$(CC) $(CC_CFLAGS) $(CFLAGS) $< -c -o $@

$(OPENMP_SPECIFIC_OBJ_FILES): %.o: %.c
	@printf '\tCC $<\n'
	@$(CC) -fopenmp $(CC_CFLAGS) $(CFLAGS) $< -c -o $@
 
$(CUDA_CU_SPECIFIC_OBJ_FILES): %.o: %.cu
	@printf '\tCC $<\n'
	@$(NVCC) $(NVCC_CFLAGS) -Xcompiler="$(CFLAGS) $(NVCC_CC_CFLAGS)" -c $< -o $@

cppcheck:
	cppcheck \
		--enable=all \
		--disable=unusedFunction,missingInclude \
		--library=posix \
		--platform=unix64 \
		--std=c11 \
		--force \
		--language=c  \
		-i src/matrix/matrix-market/mmio.c \
		--suppress=*:src/matrix/matrix-market/mmio.h \
		--suppress=*:/usr/* \
		src \
		include \
		-Iinclude \
		-I/usr/include \
		-I/usr/local/include \
		-I/usr/include/x86_64-linux-gnu \
		-I/usr/lib/gcc/x86_64-linux-gnu/13/include \
		--quiet

help:
	@echo ----help----
	@echo Valid targets: all clean common serial openmp cuda clean-common clean-serial clean-cuda clean-openmp help
	@echo Valid BUILD_TYPEs: $(valid_build_types)