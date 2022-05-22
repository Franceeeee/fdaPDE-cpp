// scans the whole mesh searching for the element containing the given point
template <unsigned int M, unsigned int N>
std::shared_ptr<Element<M, N>> BruteForce<M, N>::search(const SVector<N>& point) {
  
  for(const auto& element : mesh_){
    if(element->contains(point))
      return element;
  }

  // no element in mesh found
  return std::shared_ptr<Element<M,N>>();
}
