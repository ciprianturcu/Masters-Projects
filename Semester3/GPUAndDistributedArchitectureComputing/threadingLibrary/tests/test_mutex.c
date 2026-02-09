#include "thread.h"
#include "mutex.h"
#include <stdio.h>

mutex_type *counter_mutex;
int shared_counter = 0;

void *increment_thread(void *arg) {
    int id = *(int *)arg;
    
    for (int i = 0; i < 5; i++) {
        mutex_lock(counter_mutex);

        int old_value = shared_counter;
        printf("[thread %d] read counter: %d\n", id, old_value);

        for (volatile int j = 0; j < 50000000; j++);

        shared_counter = old_value + 1;
        printf("[thread %d] wrote counter: %d\n", id, shared_counter);

        mutex_unlock(counter_mutex);

        for (volatile int j = 0; j < 50000000; j++);
    }
    
    return NULL;
}

int main() {
    printf("mutex test\n");
    printf("starting counter: %d\n\n", shared_counter);

    if (mutex_init(&counter_mutex) != 0) {
        fprintf(stderr, "failed to initialize mutex\n");
        return 1;
    }

    thread_type t1, t2, t3;
    int id1 = 1, id2 = 2, id3 = 3;
    
    thread_create(&t1, increment_thread, &id1);
    thread_create(&t2, increment_thread, &id2);
    thread_create(&t3, increment_thread, &id3);

    thread_join(t1, NULL);
    thread_join(t2, NULL);
    thread_join(t3, NULL);
    
    printf("\nfinal counter: %d\n", shared_counter);
    printf("expected 15\n");

    mutex_destroy(counter_mutex);
    
    return 0;
}