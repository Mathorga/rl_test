#include <stdio.h>
#include <stdint.h>

void main() {
  uint64_t sum = 0;
  for (uint64_t i = 0; i < 1e10; i++) {
    sum += i;
  }
  printf("%llu\n", sum);
}