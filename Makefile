DEBUG = 0

CC = gcc

ifeq ($(DEBUG), 1)
	CFLAGS = -Og -g
else
	CFLAGS = -Wall -Wextra -Ofast
endif

LDFLAGS = -lm

test:
	$(CC) -o ML ML.c $(CFLAGS) $(LDFLAGS)
