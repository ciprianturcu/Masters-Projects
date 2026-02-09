#include "thread.h"
#include "mutex.h"
#include <stdio.h>

mutex_type *mutex_a;
mutex_type *mutex_b;

volatile int thread1_ready = 0;
volatile int thread2_ready = 0;

void *thread1_func(void *arg) {
    (void)arg;
    
    printf("[thread 1] attempting to lock mutex A\n");
    mutex_lock(mutex_a);
    printf("[thread 1] acquired mutex A\n");
    
    thread1_ready = 1;

    //busy wait to create the deadlock 
    while (!thread2_ready) {
    }

    printf("[thread 1] attempting to lock mutex B\n");
    mutex_lock(mutex_b);
    printf("[thread 1] acquired mutex B\n");
    
    mutex_unlock(mutex_b);
    mutex_unlock(mutex_a);
    
    return NULL;
}

void *thread2_func(void *arg) {
    (void)arg;
    
    printf("[thread 2] attempting to lock mutex B\n");
    mutex_lock(mutex_b);
    printf("[thread 2] acquired mutex B\n");
    
    thread2_ready = 1;

    //busy wait to create the deadlock 
    while (!thread1_ready) {
    }

    printf("[thread 2] attempting to lock mutex A\n");
    mutex_lock(mutex_a);
    printf("[thread 2] acquired mutex A\n");
    
    mutex_unlock(mutex_a);
    mutex_unlock(mutex_b);
    
    return NULL;
}

int main() {
    printf("deadlock detection test\n");

    mutex_init(&mutex_a);
    mutex_init(&mutex_b);

    thread_type t1, t2;
    
    thread_create(&t1, thread1_func, NULL);
    thread_create(&t2, thread2_func, NULL);

    thread_join(t1, NULL);
    thread_join(t2, NULL);

    mutex_destroy(mutex_a);
    mutex_destroy(mutex_b);

    return 0;
}