#ifndef MUTEX_H
#define MUTEX_H

#include "scheduler.h"

typedef struct mutex {
    int locked;
    thread_type owner;
    thread_queue waiting_queue;
} mutex_type;

int mutex_init(mutex_type** mutex);
int mutex_lock(mutex_type* mutex);
int mutex_unlock(mutex_type* mutex);
int mutex_destroy(mutex_type* mutex);

#endif
