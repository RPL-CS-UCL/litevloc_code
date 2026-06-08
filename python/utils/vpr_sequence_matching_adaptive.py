#! /usr/bin/env python

import numpy as np

if __package__:
    from .vpr_sequence_matching import PlaceRecognitionSeqMatching
else:
    from vpr_sequence_matching import PlaceRecognitionSeqMatching

class PlaceRecognitionSeqMatchingAdaptive(PlaceRecognitionSeqMatching):
	def __init__(self, seqLen):
		super().__init__(seqLen)

		# Base sequence parameters
		self.max_seq_len = seqLen  # Maximum sequence length to try
		self.min_seq_len = 4       # Minimum sequence length (adjust based on dataset)
		self.len_step = 2          # Step size for length reduction
		self.lambda_len = 0.1      # Weight for length vs cost tradeoff

	def match(self, query_descs, recall_values=1):
		"""Main entry point for sequence matching"""
		if query_descs.shape[0] < self.max_seq_len:
			return self._fallback_match(query_descs[-1, :].reshape(1, -1), recall_values)

		# Precompute integral image for fast window sum calculation
		D_all = self.compute_diff_matrix(query_descs)
		if self.enhance:
			D_all = self._enhance_contrast(D)

		self.N, self.L = D_all.shape
		best_score, best_result = 0.0, None

		# Multi-length search from longest to shortest
		for seq_len in range(self.max_seq_len, self.min_seq_len-1, -self.len_step):
			D = D_all[:, -seq_len:]
			# Calculate scores for current sequence length
			template_scores, template_velocities = \
				self._score_ref_templates(D, seq_len)
			current_preds, current_pred, current_mu = \
				self._locate_best_match(template_scores, template_velocities, seq_len, recall_values)
			
			# Combined score considering both matching quality and sequence length
			combined_score = (self.MAX_DIST - current_mu) - self.lambda_len * (1.0 / seq_len)
			if combined_score > best_score:
				best_score = combined_score
				best_result = (current_preds, current_pred, self.MAX_DIST - current_mu, seq_len)

		if best_result is not None:
			return best_result[:3]
		else:
			return self._fallback_match(query_descs[-1, :].reshape(1, -1), recall_values)
