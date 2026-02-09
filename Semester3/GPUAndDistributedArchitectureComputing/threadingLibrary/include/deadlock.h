#ifndef DEADLOCK_H
#define DEADLOCK_H

#include "scheduler.h"

void deadlock_detection_init(void);
void deadlock_detect_and_report(void);

#endif