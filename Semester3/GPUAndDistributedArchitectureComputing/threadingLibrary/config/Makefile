CC = gcc
CFLAGS = -Wall -Wextra -g -std=c11 -D_XOPEN_SOURCE=700 -Iinclude
LDFLAGS =

SRC_DIR = src
INC_DIR = include
TEST_DIR = tests
BUILD_DIR = build

SRCS = $(SRC_DIR)/thread.c $(SRC_DIR)/scheduler.c $(SRC_DIR)/mutex.c $(SRC_DIR)/deadlock.c $(SRC_DIR)/condvar.c
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
LIB = $(BUILD_DIR)/libthread.a

TESTS = test_threads test_preemption test_mutex test_deadlock test_condvar test_full_no_deadlock
TEST_BINS = $(addprefix $(BUILD_DIR)/,$(TESTS))

all: $(BUILD_DIR) $(LIB) $(TEST_BINS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_threads: $(BUILD_DIR)/test_threads.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_preemption: $(BUILD_DIR)/test_preemption.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_mutex: $(BUILD_DIR)/test_mutex.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_deadlock: $(BUILD_DIR)/test_deadlock.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_condvar: $(BUILD_DIR)/test_condvar.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_full_no_deadlock: $(BUILD_DIR)/test_full_no_deadlock.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run_test_threads: $(BUILD_DIR)/test_threads
	./$(BUILD_DIR)/test_threads

run_test_preemption: $(BUILD_DIR)/test_preemption
	./$(BUILD_DIR)/test_preemption

run_test_mutex: $(BUILD_DIR)/test_mutex
	./$(BUILD_DIR)/test_mutex

run_test_deadlock: $(BUILD_DIR)/test_deadlock
	./$(BUILD_DIR)/test_deadlock

run_test_condvar: $(BUILD_DIR)/test_condvar
	./$(BUILD_DIR)/test_condvar

run_test_full_no_deadlock: $(BUILD_DIR)/test_full_no_deadlock
	./$(BUILD_DIR)/test_full_no_deadlock

test: $(TEST_BINS)
	@echo "Running all tests..."
	@for test in $(TEST_BINS); do \
		echo "\n=== Running $$test ==="; \
		./$$test || exit 1; \
	done

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean test run_test_threads run_test_preemption run_test_mutex run_test_deadlock run_test_condvar run_test_full_no_deadlock