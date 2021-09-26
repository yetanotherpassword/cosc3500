// COSC3500, Semester 2, 2021
// Assignment 2
// Header file for the random number utilities

#if !defined(COSC3500_RANDUTIL_H)
#define COSC3500_RANDUTIL_H

#include <vector>
#include <random>
#include <initializer_list>

namespace randutil
{

//
// Functions to deal with setting / getting the random number seed
//

// seed the generator from 256 bits of 'cryptographically secure' random numbers
void seed();

// seed from a single integer
void seed(unsigned s);

// seed from an array
void seed(std::vector<unsigned> const& s);

// seed from a list
template <typename T>
void seed(std::initializer_list<T> s);

// returns the seed that was previously set
std::vector<unsigned> get_seed();

//
// Functions for random numbers
//

// A random generator for 'cryptographically secure' random numbers.
// Potentially slow, so don't use these were pseudo-random numbers will do.
// The seed() function uses this generator to seed the Mersenne Twister.
unsigned crypto_rand();

// a random unsigned integer
unsigned rand_u();

// a random integer in the closed interval [Min, Max]
int rand_int(int Min, int Max);

// returns a real number in the range [0,1)
double rand();

// returns a normal distributed real number, with mean 0, standard deviation 1
double randn();

} // namespace randutil

// include the inlined function definitions
#include "randutil.ipp"

#endif
