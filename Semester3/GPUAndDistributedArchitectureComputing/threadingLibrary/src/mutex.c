#include "mutex.h"
#include "scheduler.h"
#include <stdio.h>
#include <stdlib.h>

int mutex_init(mutex_type **mutex) {
    if (!mutex) {
        return -1;
    }
    
    *mutex = (mutex_type *)malloc(sizeof(mutex_type));
    if (!*mutex) {
        return -1;
    }
    
    (*mutex)->locked = 0;
    (*mutex)->owner = -1;
    queue_init(&(*mutex)->waiting_queue);
    
    return 0;
}

int mutex_lock(mutex_type *mutex) {
    if (!mutex) {
        return -1;
    }
    
    while (1) {
        preempt_disable();
        
        thread_control_block *current = scheduler.current_thread;
        
        if (mutex->locked == 0 || mutex->owner == current->tid) {
            mutex->locked = 1;
            mutex->owner = current->tid;
            current->waiting_on_mutex = NULL;
            preempt_enable();
            return 0;
        }
        
        current->state = T_BLOCKED;
        
        int already_waiting = 0;
        thread_control_block *check = mutex->waiting_queue.head;
        while (check) {
            if (check == current) {
                already_waiting = 1;
                break;
            }
            check = check->next_thread;
        }
        
        if (!already_waiting) {
            queue_push(&mutex->waiting_queue, current);
            current->waiting_on_mutex = mutex;
        }
        
        preempt_enable();
        schedule();
    }
}

int mutex_unlock(mutex_type *mutex) {
    if (!mutex) {
        return -1;
    }
    
    preempt_disable();

    thread_control_block *current = scheduler.current_thread;

    if (mutex->owner != current->tid) {
        preempt_enable();
        fprintf(stderr, "Error: Thread %d tried to unlock mutex owned by thread %d\n",
                current->tid, mutex->owner);
        return -1;
    }
    
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
    
    return 0;
}

int mutex_destroy(mutex_type *mutex) {
    if (!mutex) {
        return -1;
    }
    
    preempt_disable();

    if (mutex->locked) {
        preempt_enable();
        fprintf(stderr, "Error: Attempting to destroy locked mutex\n");
        return -1;
    }
    
    if (mutex->waiting_queue.head != NULL) {
        preempt_enable();
        fprintf(stderr, "Error: Attempting to destroy mutex with waiting threads\n");
        return -1;
    }
    
    free(mutex);
    
    preempt_enable();
    
    return 0;
}