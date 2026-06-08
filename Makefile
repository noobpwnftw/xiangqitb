CXX      ?= g++
CXX_IS_GCC := $(shell $(CXX) -dM -E -x c++ /dev/null | awk '/__GNUC__/ { gcc = 1 } /__clang__/ { clang = 1 } END { print gcc && !clang }')

OPT_FLAGS ?= -O3 -DNDEBUG -flto=auto
COMMON_FLAGS ?= $(OPT_FLAGS) -march=native -Wall -Isrc -Ilib
CXXFLAGS ?= -std=c++17 $(COMMON_FLAGS)
CFLAGS   ?= $(COMMON_FLAGS)
LDFLAGS  ?= -pthread

TARGET ?= xiangqitb
OBJDIR ?= obj

# Command echo: quiet by default, `make V=1` prints the full command lines.
V ?= 0
ifeq ($(V),0)
Q := @
E := @echo
else
Q :=
E := @:
endif
# Suppress the sub-make's Entering/Leaving lines too, but only when quiet.
NPD := $(if $(Q),--no-print-directory)

ifeq ($(CXX_IS_GCC),1)
CXXFLAGS += -Wno-interference-size -Wno-class-memaccess
endif

ifeq ($(shell uname -m),aarch64)
CXXFLAGS += -mno-outline-atomics
CFLAGS   += -mno-outline-atomics
endif

COMMON_C := \
  $(wildcard lib/lz4/*.c) \
  $(wildcard lib/LZMA/*.c) \
  $(wildcard lib/zstd/common/*.c) \
  $(wildcard lib/zstd/compress/*.c) \
  $(wildcard lib/zstd/dictBuilder/*.c)

COMMON_CXX := \
  $(wildcard src/chess/*.cpp) \
  $(wildcard src/egtb/*.cpp) \
  $(wildcard src/util/*.cpp)

HDRS := $(shell find src lib -name '*.h' -o -name '*.hpp' 2>/dev/null)
LIB_HDRS := $(shell find lib -name '*.h' -o -name '*.hpp' 2>/dev/null)

COMMON_C_OBJ := $(COMMON_C:%.c=$(OBJDIR)/%.o)

$(OBJDIR)/%.o: %.c $(LIB_HDRS) Makefile
	@mkdir -p $(dir $@)
	$(E) "  CC      $<"
	$(Q)$(CXX) $(CFLAGS) -x c -c $< -o $@

ifeq ($(CXX_IS_GCC),1)
$(OBJDIR)/lib/LZMA/LzmaEnc.o: CFLAGS += -Wno-dangling-pointer
endif

.PHONY: all debug clean
all: $(TARGET)

$(TARGET): src/main.cpp $(COMMON_CXX) $(COMMON_C_OBJ) $(HDRS) Makefile
	$(E) "  CXXLD   $(TARGET)"
	$(Q)$(CXX) $(CXXFLAGS) \
	    -x c++ src/main.cpp $(COMMON_CXX) \
	    -x none $(COMMON_C_OBJ) \
	    $(LDFLAGS) -o $@

# Asserts enabled (no -DNDEBUG), unoptimized, with debug info. Kept in its own
# object dir and under its own name so it never collides with the release build.
debug:
	$(Q)$(MAKE) $(NPD) OPT_FLAGS="-O0 -g" OBJDIR=obj-debug TARGET=$(TARGET)-debug

clean:
	$(Q)rm -rf obj obj-debug
	$(Q)rm -f xiangqitb xiangqitb-debug
