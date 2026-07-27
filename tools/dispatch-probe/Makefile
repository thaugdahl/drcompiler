CC      ?= cc
CFLAGS  ?= -O2 -march=native -Wall -Wextra
TARGET   = dispatch_probe

$(TARGET): dispatch_probe.c
	$(CC) $(CFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
