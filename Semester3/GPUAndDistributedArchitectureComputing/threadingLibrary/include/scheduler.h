#pragma once
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <ucontext.h>

#define STACK_SIZE (128 * 1024)
#define MAX_THREADS 1024

typedef int thread_type;

typedef enum {
	T_READY,
	T_RUNNING,
	T_BLOCKED,
	T_TERMINATED
}thread_state;

typedef struct tcb {
	thread_type tid;
	ucontext_t context;
	void* stack;
	thread_state state;
	void* retval;

	int joinable;
	struct tcb* joining_thread;
	struct tcb* next_thread;
	struct mutex* waiting_on_mutex;
} thread_control_block;

typedef struct {
	thread_control_block* head;
	thread_control_block* tail;
} thread_queue;

typedef struct ss {
	thread_control_block* current_thread;
	thread_queue ready_queue;
	thread_control_block* all_threads[MAX_THREADS];
	int thread_count;
} thread_scheduler;

extern thread_scheduler scheduler;

void scheduler_init(void);
void schedule(void);
thread_control_block *get_thread_by_id(thread_type thread_id);
thread_control_block *create_tcb(void);
void add_thread_to_scheduler(thread_control_block *new_thread);

void queue_init(thread_queue *queue);
void queue_push(thread_queue *queue, thread_control_block *thread);
thread_control_block* queue_pop(thread_queue* queue);

void preempt_init(void);
void preempt_disable(void);
void preempt_enable(void);


#endif



