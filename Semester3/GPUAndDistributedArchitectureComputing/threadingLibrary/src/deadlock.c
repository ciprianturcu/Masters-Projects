#include "deadlock.h"
#include "scheduler.h"
#include "mutex.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>

static void sigquit_handler(int signum) {
    (void)signum;
    
    printf("\ndeadlock detection triggered (SIGQUIT)\n");
    
    deadlock_detect_and_report();
}

void deadlock_detection_init(void) {
    struct sigaction sa;
    
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigquit_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    
    if (sigaction(SIGQUIT, &sa, NULL) == -1) {
        perror("sigaction (SIGQUIT)");
        exit(1);
    }
    
    printf("deadlock detection succesfuly initialized - Press Ctrl+\\ to check for deadlocks\n\n");
}

static int has_cycle_from(thread_type start_tid, thread_type current_tid, int *visited, int *rec_stack) {
    if (!visited[current_tid]) {
        visited[current_tid] = 1;
        rec_stack[current_tid] = 1;
        
        thread_control_block *current_thread = get_thread_by_id(current_tid);

        if (current_thread && current_thread->waiting_on_mutex) {
            mutex_type *mutex = current_thread->waiting_on_mutex;

            thread_type owner_tid = mutex->owner;

            if (owner_tid == -1) {
                rec_stack[current_tid] = 0;
                return 0;
            }

            if (rec_stack[owner_tid]) {
                return 1;
            }

            if (!visited[owner_tid]) {
                if (has_cycle_from(start_tid, owner_tid, visited, rec_stack)) {
                    return 1;
                }
            }
        }
    }
    
    rec_stack[current_tid] = 0;
    return 0;
}

static int detect_deadlock_cycle(thread_type *cycle_threads, int *cycle_count) {
    int visited[MAX_THREADS];
    int rec_stack[MAX_THREADS];
    
    memset(visited, 0, sizeof(visited));
    memset(rec_stack, 0, sizeof(rec_stack));

    for (int i = 0; i < MAX_THREADS; i++) {
        thread_control_block *thread = scheduler.all_threads[i];
        
        if (thread && thread->state == T_BLOCKED && thread->waiting_on_mutex) {
            if (has_cycle_from(thread->tid, thread->tid, visited, rec_stack)) {
                *cycle_count = 0;
                for (int j = 0; j < MAX_THREADS; j++) {
                    if (rec_stack[j]) {
                        cycle_threads[*cycle_count] = j;
                        (*cycle_count)++;
                    }
                }
                return 1;
            }
        }
    }
    
    return 0;
}

static void print_thread_state(thread_control_block *thread) {
    printf("  thread ID: %d\n", thread->tid);
    
    switch (thread->state) {
        case T_READY:
            printf("  State: READY\n");
            break;
        case T_RUNNING:
            printf("  State: RUNNING\n");
            break;
        case T_BLOCKED:
            printf("  State: BLOCKED\n");
            break;
        case T_TERMINATED:
            printf("  State: TERMINATED\n");
            break;
    }
    
    if (thread->waiting_on_mutex) {
        mutex_type *mutex = thread->waiting_on_mutex;
        printf("  waiting on: mutex@%p (owned by thread %d)\n", 
               (void *)mutex, mutex->owner);
    }
    
    printf("\n");
}

void deadlock_detect_and_report(void) {
    preempt_disable();
    
    thread_type cycle_threads[MAX_THREADS];
    int cycle_count = 0;
    
    printf("state of current threads:\n");
    
    int blocked_count = 0;

    for (int i = 0; i < MAX_THREADS; i++) {
        thread_control_block *thread = scheduler.all_threads[i];
        
        if (thread) {
            print_thread_state(thread);
            
            if (thread->state == T_BLOCKED) {
                blocked_count++;
            }
        }
    }
    
    if (blocked_count == 0) {
        printf("no blocked threads - no deadlock possible.\n");
        preempt_enable();
        return;
    }
    
    printf("analyzing wait-for graph\n\n");

    if (detect_deadlock_cycle(cycle_threads, &cycle_count)) {
        printf("detected a deadlock\n");
        printf("the threads involved in deadlock cycle:\n");
        
        for (int i = 0; i < cycle_count; i++) {
            thread_control_block *thread = get_thread_by_id(cycle_threads[i]);
            if (thread) {
                printf("  → thread %d", thread->tid);
                
                if (thread->waiting_on_mutex) {
                    mutex_type *mutex = thread->waiting_on_mutex;
                    printf(" waiting for mutex@%p (owned by thread %d)",
                           (void *)mutex, mutex->owner);
                }
                printf("\n");
            }
        }
        
        printf("\nDeadlock cycle detected with %d threads.\n", cycle_count);
    } else {
        printf("✓ No deadlock detected.\n");
        printf("  (%d blocked thread(s), but no circular wait)\n", blocked_count);
    }
    
    preempt_enable();
}