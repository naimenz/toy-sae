
import torch


def get_dictionary_score(weights, ground_truth_vectors) -> float:
    """Compute a score for how well the weights have learned the ground truth
    dense vectors.
    
    We compute the score by:
        - for each weight vector, compute dot products with all ground truth vectors
        - pick the ground truth vector with the highest dot product
        - add the dot product with that vector to the sum
        - normalize by the number of vectors

    Args:
        weights: the weights of the SAE
        ground_truth_vectors: the ground truth dense vectors
    """

    score = 0.
    for w in weights:
        # we normalize the weight vector just to make the dot products easier to interpret
        normed_w = w / torch.norm(w)
        best_dot = -float("inf")
        best_gt = None
        for gt in ground_truth_vectors:
            current_dot = torch.dot(normed_w, gt).item()
            if current_dot > best_dot:
                best_dot = current_dot
                best_gt = gt
        score += best_dot

    normed_score = score / len(weights)
    return normed_score



        