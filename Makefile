DEBUG = 0

CC = gcc

ifeq ($(DEBUG), 1)
	CFLAGS = -Og -g
else
	CFLAGS = -Wall -Wextra -Ofast
endif

LDFLAGS = -lm

test:
	$(CC) -o CCeptron CCeptron.c $(CFLAGS) $(LDFLAGS)
