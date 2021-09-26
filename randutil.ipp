// COSC3500, Semester 2, 2021
// Assignment 2
// Inline Implementation file for the random number utilities

namespace randutil
{

namespace detail
{
   extern std::uniform_real_distribution<double> UniformDist;
   extern std::normal_distribution<double> NormalDist;
   extern std::mt19937 u_rand;
} // namespace detail

inline
unsigned rand_u()
{
   return detail::u_rand();
}

inline
int rand_int(int Min, int Max)
{
   return int(std::floor(randutil::rand() * (Max-Min+1))) + Min;
}

inline
double rand()
{
   return detail::UniformDist(detail::u_rand);
}

inline
double randn()
{
   return detail::NormalDist(detail::u_rand);
}

template <typename T>
void seed(std::initializer_list<T> s)
{
   seed(std::vector<unsigned>(s));
}

} // namespace randutil
