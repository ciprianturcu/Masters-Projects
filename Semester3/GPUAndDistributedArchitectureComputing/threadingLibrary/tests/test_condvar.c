#include "thread.h"
#include "mutex.h"
#include "condvar.h"
#include <stdio.h>

#define BUFFER_SIZE 5

int buffer[BUFFER_SIZE];
int count = 0;
int in = 0;
int out = 0;

mutex_type *mutex;
condvar_type *cond_full;
condvar_type *cond_empty;

void *producer(void *arg) {
    int id = *(int *)arg;
    
    for (int i = 0; i < 10; i++) {
        int item = id * 100 + i;
        
        mutex_lock(mutex);

        while (count == BUFFER_SIZE) {
            printf("[producer %d] buffer full, waiting\n", id);
            cond_wait(cond_empty, mutex);
        }

        buffer[in] = item;
        in = (in + 1) % BUFFER_SIZE;
        count++;

        printf("[producer %d] produced: %d (count=%d)\n", id, item, count);

        cond_signal(cond_full);

        mutex_unlock(mutex);

        for (volatile long j = 0; j < 50000000; j++);
    }
    
    printf("[producer %d] done\n", id);
    return NULL;
}

void *consumer(void *arg) {
    int id = *(int *)arg;
    
    for (int i = 0; i < 10; i++) {
        mutex_lock(mutex);

        while (count == 0) {
            printf("[consumer %d] buffer empty, waiting\n", id);
            cond_wait(cond_full, mutex);
        }

        int item = buffer[out];
        out = (out + 1) % BUFFER_SIZE;
        count--;

        printf("[consumer %d] consumed: %d (count=%d)\n", id, item, count);

        cond_signal(cond_empty);

        mutex_unlock(mutex);

        for (volatile long j = 0; j < 100000000; j++);
    }
    
    printf("[consumer %d] done\n", id);
    return NULL;
}

int main() {
    printf("conditional variables test - consumer-producer\n");
    printf("buffer size: %d\n\n", BUFFER_SIZE);

    mutex_init(&mutex);
    cond_init(&cond_full);
    cond_init(&cond_empty);

    thread_type p1, p2, c1, c2;
    int id1 = 1, id2 = 2, id3 = 3, id4 = 4;

    thread_create(&p1, producer, &id1);
    thread_create(&p2, producer, &id2);
    thread_create(&c1, consumer, &id3);
    thread_create(&c2, consumer, &id4);

    thread_join(p1, NULL);
    thread_join(p2, NULL);
    thread_join(c1, NULL);
    thread_join(c2, NULL);

    printf("\ntest complete\n");
    printf("final buffer count: %d (should be 0)\n", count);

    cond_destroy(cond_full);
    cond_destroy(cond_empty);
    mutex_destroy(mutex);

    return 0;
}