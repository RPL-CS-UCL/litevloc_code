import numpy as np
import torch
from typing import Tuple

class PlaceRecognitionTopologicalFilter:
    '''
    Adapted from https://github.com/mingu6/ProbFiltersVPR/blob/master/src/models/TopologicalFilter.py
    '''
    def __init__(self, db_descriptors, delta=5, window_lower=-2, window_upper=10, recall_values=5):        
        # get map descriptors
        self.db_descriptors = db_descriptors

        # initialize hidden states and obs likelihood parameters
        self.delta = delta
        self.lambda1 = 0.0
        self.belief = None

        # parameters for the transition matrix
        self.window_lower = window_lower
        self.window_upper = window_upper
        self.window_size = int((window_upper - window_lower) / 2)
        self.transition = np.ones(window_upper - window_lower)

        self.recall_values = recall_values

    def initialize_model(self, query_desc):
        '''
        Initialize the belief distribution from the distance of the
        query descriptor to the database descriptors
        
        TODO: Alternatively the initial belief 
              could be concentrated on the start node
        '''
        # Distance between query descriptor and database descriptors
        dists = np.sqrt(2 - 2 * np.dot(self.db_descriptors, query_desc))
        
        # Init for lambda1
        descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
        self.lambda1 = np.log(self.delta) / (
            descriptor_quantiles[1] - descriptor_quantiles[0]
        )

        # Init for belief distribution
        self.belief = np.exp(-self.lambda1 * dists)
        self.belief /= self.belief.sum()

        recall_preds = np.argsort(self.belief)[-5:][::-1]

        pred = np.argmax(self.belief)
        score = self.belief[pred]

        return recall_preds, pred, score

    def obs_lhood(self, descriptor: np.ndarray) -> np.ndarray:
        '''Observation likelihood of the query descriptor'''
        vsim = np.exp(
            -self.lambda1 * np.sqrt(2 - 2 * np.dot(self.db_descriptors, descriptor))
        )
        return vsim

    def match(self, query_desc: torch.Tensor) -> Tuple[int, float]:
        '''
        Match the query image to the topological map.

        Runs a prediction step followed by a measurement step:
        - Prediction: Propagate belief mass using the transition model
        - Measurement: Update belief mass using the observation likelihood

        After the process, the map node with the highest probability is
        returned as the subgoal.

        Returns:
        - proposal: the index of the subgoal
        - score: the probability of the subgoal
        '''
        # Calculate the indices of the valid data within the convolution output
        # for the implementation of the prediction step / transition model
        w_l = self.window_lower
        if w_l < 0:
            # In the prediction step, node acquires belief mass
            # from nodes both 'before' and 'after' it in the topomap
            conv_ind_l, conv_ind_h = np.abs(w_l), len(self.belief) + np.abs(w_l)
            bel_ind_l, bel_ind_h = 0, len(self.belief)
        else:
            # Only from nodes 'after' it (and node itself)
            conv_ind_l, conv_ind_h = 0, len(self.belief) - w_l
            bel_ind_l, bel_ind_h = w_l, len(self.belief)

        # Apply transition matrix as a convolution
        # first add symmetric padding to remove edge effects
        # (without it first and last node never reach high values)
        belief_pad = np.pad(self.belief, len(self.transition)-1, mode='symmetric')
        conv = np.convolve( belief_pad, self.transition, mode='valid')

        # Pick the valid part of the convolution output
        self.belief[bel_ind_l:bel_ind_h] = conv[
            conv_ind_l:conv_ind_h
        ]

        if w_l > 0:
            self.belief[:w_l] = 0.0

        # observation likelihood update
        obs_lhood = self.obs_lhood(query_desc)
        self.belief *= obs_lhood
        self.belief /= self.belief.sum()
        
        recall_preds = np.argsort(self.belief)[-5:][::-1]
        
        # Argmax of the belief
        pred = np.argmax(self.belief)
        score = self.belief[pred]
       
        return recall_preds, pred, score