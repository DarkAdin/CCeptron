DEBUG = 0

CC = gcc

ifeq ($(DEBUG), 1)
	CFLAGS = -Og -g3
else
    CFLAGS = -Wall -Wextra -Ofast -funroll-loops
endif

LDFLAGS = -lm

all:
	$(CC) -o CCeptron CCeptron.c $(CFLAGS) $(LDFLAGS)
