#include "thread.h"
#include "scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>

static void thread_wrapper(void* (*start_routine)(void*), void* arg) {
    void* retval = start_routine(arg);
    thread_exit(retval);
}

void thread_init(void) {
    static int initialized = 0;
    if (initialized) return;
    initialized = 1;

    scheduler_init();
}

int thread_create(thread_type* thread, void* (*start_routine)(void*), void* arg) {
    thread_init();

    thread_control_block* new_thread = create_tcb();
    if (!new_thread) {
        return -1;
    }

    if (getcontext(&new_thread->context) == -1) {
        free(new_thread->stack);
        free(new_thread);
        return -1;
    }

    new_thread->context.uc_stack.ss_sp = new_thread->stack;
    new_thread->context.uc_stack.ss_size = STACK_SIZE;
    new_thread->context.uc_stack.ss_flags = 0;
    new_thread->context.uc_link = NULL;

    makecontext(&new_thread->context, (void (*)())thread_wrapper, 2, start_routine, arg);

    add_thread_to_scheduler(new_thread);

    *thread = new_thread->tid;
    return 0;
}

void thread_exit(void* retval) {
    thread_control_block* current = scheduler.current_thread;

    current->state = T_TERMINATED;
    current->retval = retval;

    if (current->joining_thread) {
        current->joining_thread->state = T_READY;
        queue_push(&scheduler.ready_queue, current->joining_thread);
        current->joining_thread = NULL;
    }

    schedule();

    fprintf(stderr, "error: thread_exit returned from schedule\n");
    exit(1);
}

int thread_join(thread_type thread, void** retval) {
    thread_control_block* target = get_thread_by_id(thread);

    if (!target || !target->joinable) {
        return -1;
    }

    if (target->tid == scheduler.current_thread->tid) {
        return -1;
    }

    if (target->state == T_TERMINATED) {
        if (retval) {
            *retval = target->retval;
        }
        return 0;
    }

    target->joining_thread = scheduler.current_thread;
    scheduler.current_thread->state = T_BLOCKED;

    schedule();

    if (retval) {
        *retval = target->retval;
    }

    return 0;
}

thread_type thread_self(void) {
    return scheduler.current_thread->tid;
}

void thread_yield(void) {
    thread_control_block* current = scheduler.current_thread;

    if (current->state == T_RUNNING) {
        current->state = T_READY;
        queue_push(&scheduler.ready_queue, current);
    }

    schedule();
}