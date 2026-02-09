#include "condvar.h"
#include "scheduler.h"
#include "mutex.h"
#include <stdio.h>
#include <stdlib.h>

int cond_init(condvar_type **cond) {
    if (!cond) {
        return -1;
    }
    
    *cond = (condvar_type *)malloc(sizeof(condvar_type));
    if (!*cond) {
        return -1;
    }
    
    queue_init(&(*cond)->waiting_queue);
    
    return 0;
}

int cond_wait(condvar_type *cond, mutex_type *mutex) {
    if (!cond || !mutex) {
        return -1;
    }
    
    preempt_disable();
    
    thread_control_block *current = scheduler.current_thread;

    if (mutex->owner != current->tid) {
        preempt_enable();
        fprintf(stderr, "Error: Thread %d called cond_wait without owning mutex\n",
                current->tid);
        return -1;
    }
    
    current->state = T_BLOCKED;
    queue_push(&cond->waiting_queue, current);

    thread_control_block *next = queue_pop(&mutex->waiting_queue);

    if (next) {
        mutex->owner = next->tid;
        next->state = T_READY;
        queue_push(&scheduler.ready_queue, next);
    } else {
        mutex->locked = 0;
        mutex->owner = -1;
    }

    preempt_enable();

    schedule();

    mutex_lock(mutex);
    
    return 0;
}

int cond_signal(condvar_type *cond) {
    if (!cond) {
        return -1;
    }
    
    preempt_disable();

    thread_control_block *thread = queue_pop(&cond->waiting_queue);
    
    if (thread) {
        thread->state = T_READY;
        queue_push(&scheduler.ready_queue, thread);
    }
    
    preempt_enable();
    
    return 0;
}

int cond_broadcast(condvar_type *cond) {
    if (!cond) {
        return -1;
    }
    
    preempt_disable();

    thread_control_block *thread;
    while ((thread = queue_pop(&cond->waiting_queue)) != NULL) {
        thread->state = T_READY;
        queue_push(&scheduler.ready_queue, thread);
    }
    
    preempt_enable();
    
    return 0;
}

int cond_destroy(condvar_type *cond) {
    if (!cond) {
        return -1;
    }
    
    preempt_disable();

    if (cond->waiting_queue.head != NULL) {
        preempt_enable();
        fprintf(stderr, "Error: Attempting to destroy condition variable with waiting threads\n");
        return -1;
    }
    
    free(cond);
    
    preempt_enable();
    
    return 0;
}