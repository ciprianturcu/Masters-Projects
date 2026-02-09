#include "scheduler.h"
#include "deadlock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>

#define INTERVAL_MS 10

thread_scheduler scheduler;

static void timer_handler(int signum) {
	(void)signum;

	thread_control_block* current = scheduler.current_thread;

	if (current->state == T_RUNNING) {
		if (scheduler.ready_queue.head == NULL) {
			return;
		}
		current->state = T_READY;
		queue_push(&scheduler.ready_queue, current);
		schedule();
	}
}

void preempt_init(void) {
	struct sigaction sa;
	struct itimerval timer;

	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = timer_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART;

	if (sigaction(SIGALRM, &sa, NULL) == -1) {
		perror("sigaction");
		exit(1);
	}

	timer.it_value.tv_sec = 0;
	timer.it_value.tv_usec = INTERVAL_MS * 1000;
	timer.it_interval.tv_sec = 0;
	timer.it_interval.tv_usec = INTERVAL_MS * 1000;

	if (setitimer(ITIMER_REAL, &timer, NULL) == -1) {
		perror("setitimer");
		exit(1);
	}
}

void preempt_disable(void) {
	sigset_t mask;
	sigemptyset(&mask);
	sigaddset(&mask, SIGALRM);
	sigprocmask(SIG_BLOCK, &mask, NULL);
}

void preempt_enable(void) {
	sigset_t mask;
	sigemptyset(&mask);
	sigaddset(&mask, SIGALRM);
	sigprocmask(SIG_UNBLOCK, &mask, NULL);
}

void queue_init(thread_queue* queue) {
	queue->head = NULL;
	queue->tail = NULL;
}

void queue_push(thread_queue* queue, thread_control_block* thread) {
	thread->next_thread = NULL;
	if (queue->tail) {
		queue->tail->next_thread = thread;
		queue->tail = thread;
	}
	else {
		queue->head = queue->tail = thread;
	}
}

thread_control_block* queue_pop(thread_queue* queue) {
	if (!queue->head)
		return NULL;
	thread_control_block* thread = queue->head;
	queue->head = queue->head->next_thread;

	if (!queue->head)
		queue->tail = NULL;

	thread->next_thread = NULL;
	return thread;
}

void scheduler_init(void) {
	memset(&scheduler, 0, sizeof(thread_scheduler));
	queue_init(&scheduler.ready_queue);

	for (int i = 0; i < MAX_THREADS; i++)
		scheduler.all_threads[i] = NULL;

	thread_control_block *main_thread = (thread_control_block*)malloc(sizeof(thread_control_block));
	if (!main_thread) {
		fprintf(stderr, "failed to allocate main thread TCB\n");
		exit(1);
	}

	main_thread->tid = 0;
	main_thread->state = T_RUNNING;
	main_thread->stack = NULL;
	main_thread->retval = NULL;
	main_thread->joinable = 0;
	main_thread->joining_thread = NULL;
	main_thread->next_thread = NULL;
	main_thread->waiting_on_mutex = NULL;

	if (getcontext(&main_thread->context) == -1) {
		perror("getcontext - main_thread");
		exit(1);
	}

	scheduler.current_thread = main_thread;
	scheduler.all_threads[0] = main_thread;
	scheduler.thread_count = 1;

	preempt_init();
	deadlock_detection_init();
}

static int find_free_slot(void) {
	for (int i = 0; i < MAX_THREADS; i++) {
		if (scheduler.all_threads[i] == NULL) {
			return i;
		}
	}
	return -1;
}

void schedule(void) {
	thread_control_block* prev_thread = scheduler.current_thread;
	thread_control_block* next_thread;

	next_thread = queue_pop(&scheduler.ready_queue);
	if (!next_thread) {
		if (prev_thread->state == T_TERMINATED)
			exit(0);
		else
			return;
	}
	next_thread->state = T_RUNNING;
	scheduler.current_thread = next_thread;

	if (prev_thread->state == T_TERMINATED) {
		setcontext(&next_thread->context);
	}
	else {
		swapcontext(&prev_thread->context, &next_thread->context);
	}
}

thread_control_block* get_thread_by_id(thread_type tid) {
	if (tid < 0 || tid >= MAX_THREADS) {
		return NULL;
	}
	return scheduler.all_threads[tid];
}

thread_control_block* create_tcb(void) {
	if (scheduler.thread_count >= MAX_THREADS) {
		return NULL;
	}

	int free_slot = find_free_slot();
	if (free_slot == -1) {
		return NULL; 
	}

	thread_control_block* new_thread = (thread_control_block*)malloc(sizeof(thread_control_block));
	if (!new_thread)
		return NULL;
	new_thread->stack = malloc(STACK_SIZE);
	if (!new_thread->stack) {
		free(new_thread);
		return NULL;
	}
	

	new_thread->tid = free_slot;
	new_thread->state = T_READY;
	new_thread->retval = NULL;
	new_thread->joinable = 1;
	new_thread->joining_thread = NULL;
	new_thread->next_thread = NULL;
	new_thread->waiting_on_mutex=NULL;


	if (getcontext(&new_thread->context) == -1) {
		free(new_thread->stack);
		free(new_thread);
		return NULL;
	}

	return new_thread;
}

void add_thread_to_scheduler(thread_control_block* thread) {
	scheduler.all_threads[thread->tid] = thread;
	scheduler.thread_count++;
	queue_push(&scheduler.ready_queue, thread);
}