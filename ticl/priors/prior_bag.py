import torch


class BagPrior:
    def __init__(self, base_priors, prior_weights, verbose=False):
        self.base_priors = base_priors
        # let's make sure we get consistent sorting of the base priors by name
        self.prior_names = sorted(base_priors.keys())
        self.prior_weights = prior_weights
        self.verbose = verbose

    def get_batch(
        self, 
        *, 
        batch_size, 
        n_samples, 
        num_features, 
        device, 
        epoch=None, 
        single_eval_pos=None
    ):
        args = {
            'device': device, 
            'n_samples': n_samples, 
            'num_features': num_features,
            'batch_size': batch_size, 
            'epoch': epoch, 
            'single_eval_pos': single_eval_pos
        }

        weights = torch.tensor([self.prior_weights[prior_name] for prior_name in self.prior_names], dtype=torch.float)
        weights = weights / torch.sum(weights)
        batch_assignments = torch.multinomial(weights, 1, replacement=True).numpy()

        # Get the appropriate prior
        selected_prior = self.base_priors[self.prior_names[int(batch_assignments[0])]]
        
        # Check if the semantic_feature_p attribute exists and is set to zero
        semantic_feature_p = getattr(selected_prior, 'semantic_feature_p', 0.0)
        
        # Get batch from the selected prior
        prior_result = selected_prior.get_batch(**args)
        
        # Handle different return formats based on semantic_feature_p
        if semantic_feature_p <= 0.0 and len(prior_result) == 3:
            # Original format (x, y, y_)
            x, y, y_ = prior_result
            info = {}  # Create empty info dict
        else:
            # New format (x, y, y_, info)
            if len(prior_result) == 4:
                x, y, y_, info = prior_result
            else:
                # Fallback for unexpected tuple length
                x, y, y_ = prior_result[:3]
                info = {}
        
        return x.detach(), y.detach(), y_.detach(), info
