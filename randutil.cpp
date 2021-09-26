// COSC3500, Semester 2, 2021
// Assignment 2
// Implementation file for the random number utilities

#include "randutil.h"

namespace randutil
{

// A fixed seed, so we intialize the generator to a known state
std::vector<unsigned> Seed{1,2,3,4,5,6};

// some awkwardness: the mt19937 requires an l-value reference, we can't initialize it from a temporary
std::seed_seq TempInitializer(Seed.begin(), Seed.end());

namespace detail
{
   std::mt19937 u_rand(TempInitializer);
   std::uniform_real_distribution<double> UniformDist(0,1);
   std::normal_distribution<double> NormalDist;
} // namespace detail

std::random_device rd;

unsigned crypto_rand()
{
   // We used to have a mutex here, but apparantly random_device is thread_safe.
   // generally we wouldn't want to call this from multiple threads anyway.
   return rd();
}

void seed()
{
   seed({crypto_rand(), crypto_rand(), crypto_rand(), crypto_rand(),
         crypto_rand(), crypto_rand(), crypto_rand(), crypto_rand()});
}

// seed from a single integer
void seed(unsigned s)
{
   seed(std::vector<unsigned>(1,s));
}

// seed from an array
void seed(std::vector<unsigned> const& s)
{
   Seed = s;
   std::seed_seq SS(Seed.begin(), Seed.end());
   detail::u_rand.seed(SS);
}

std::vector<unsigned> get_seed()
{
   return Seed;
}

} // namespace randutil
