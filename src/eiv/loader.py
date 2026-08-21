import jax
import jax.numpy as jnp
import pynapple as nap
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class FentonHeadDirection:
    def __init__(self, filepath):
        data = pd.read_pickle(filepath)
        spikes_key = 0
        yaw_key = 1
        locomotion_key = 2

        light_fr = 0
        dark_fr = 1
        light_hf = 2
        dark_hf = 3

        self.data = {"spikes" : {
                    "fr_light": data[spikes_key][light_fr][:],
                    "fr_dark": data[spikes_key][dark_fr][:],
                    "hf_light": data[spikes_key][light_hf][:],
                    "hf_dark": data[spikes_key][dark_hf][:]
                },
                "headdir": {
                    "fr_light":  data[yaw_key][light_fr][:],
                    "fr_dark":  data[yaw_key][dark_fr][:],
                    "hf_light":  data[yaw_key][light_hf][:],
                    "hf_dark":  data[yaw_key][dark_hf][:]
                },
                "locomotion": {
                    "fr_light":  data[locomotion_key][light_fr][:],
                    "fr_dark":  data[locomotion_key][dark_fr][:],
                    "hf_light":  data[locomotion_key][light_hf][:],
                    "hf_dark":  data[locomotion_key][dark_hf][:]
                }
            }
        

    def load_data(self, animal, epoch, bin_size, filter_locomotion=False, filter_nan = True, normed = True):
        
        max_ind = jnp.min(jnp.array([jnp.array(self.data["headdir"][epoch][animal]).shape[0], self.data["spikes"][epoch][animal].shape[1]]))
        if filter_locomotion:
            X_obs = np.radians(jnp.array(self.data["headdir"][epoch][animal])[:max_ind][np.where(self.data["locomotion"][epoch][animal])[0]])
            Y = self.data["spikes"][epoch][animal][:,np.where(self.data["locomotion"][epoch][animal][:max_ind])[0]].T
            
            
            min_ind = np.min([X_obs.shape[0], Y.shape[0]])
            leftover = min_ind%bin_size
            Y = Y[:min_ind-leftover,:]
            X_obs = X_obs[:min_ind-leftover]
        else:
            X_obs = np.radians(self.data["headdir"][epoch][animal][:max_ind])
            Y = self.data["spikes"][epoch][animal][:max_ind].T
            # TODO - check on this with jose
            min_ind = np.min([X_obs.shape[0], Y.shape[0]])
            leftover = min_ind%bin_size
            Y = Y[:min_ind-leftover,:]
            X_obs = X_obs[:min_ind-leftover]
        
        X_obs = jnp.mean(X_obs.reshape(int(X_obs.shape[0]/bin_size), bin_size), axis=1)
        
        Y = jnp.mean(Y.reshape(int(Y.shape[0]/bin_size), bin_size, Y.shape[1]), axis=1)
        
        if filter_nan:
            X_obs = jnp.array(X_obs)
            Y = Y[~np.isnan(X_obs)]
            X_obs = X_obs[~np.isnan(X_obs)]
        
        print("Timesteps x Neurons: " +str(Y.shape))
        #TODO - confirm this
        if normed:
            X_obs = X_obs/(2*jnp.pi)

        return [Y, X_obs]

    def get_tuning(self, n_bins, n_neurons, spikes, headdir, return_bins = False):
        
        bins = np.linspace(np.min(headdir), np.max(headdir), n_bins+1)
        tuning_curves = np.zeros((n_neurons, n_bins))
        for i in range(n_neurons):
            digi = np.digitize(headdir, bins)
            tuning_curves[i, :] = [np.nanmean(spikes[np.where(digi==j)[0],i]) for j in range(1, n_bins+1)]
        
        if return_bins:
            return tuning_curves, bins[1:]
        else:
            return tuning_curves

class PeyracheHeadDirection:
    def __init__(self,filepath):
        self.data = nap.load_file(filepath)

    def load_data(self, epoch, bin_size, filter_nan = True, normed = True):
        spikes = self.data["units"]  # Get spike timings
        epochs = self.data["epochs"]  # Get the behavioural epochs (in this case, sleep and wakefulness)
        angle = self.data["ry"]  # Get the tracked orientation of the animal
        spikes_adn = spikes.getby_category("location")["adn"]  # Select only those units that are in ADn
        
        # TODO - make it so you can specify epoch with loader
        self.tuning_curves = nap.compute_1d_tuning_curves(
                            group=spikes_adn,
                            feature=angle,
                            nb_bins=50,
                            ep = epochs[epochs.tags == 'wake'],
                            minmax=(0, 2 * jnp.pi)
                            )

        pref_ang_sorted = jnp.argsort(jnp.array(self.tuning_curves.idxmax()))#[10:20]
        self.tuning_curves = jnp.array(self.tuning_curves)[:,pref_ang_sorted] * bin_size
        
        binned = spikes_adn.count(bin_size, ep=epochs[epochs.tags == epoch]) # bin 500ms
       
        Y = jnp.array(binned)[:,pref_ang_sorted]# sort neurons by preferred angle
        
        X_obs = angle.bin_average(bin_size)
        
        min_ind = np.min([X_obs.shape[0], Y.shape[0]])
        
        Y = Y[:min_ind,:]
        X_obs = X_obs[:min_ind]
        #jnp.array([jnp.mean(angle_split[i]) for i in range(len(angle_bins))])
        if epoch == "wake":
            if filter_nan:
                X_obs = jnp.array(X_obs)
                Y = Y[:len(X_obs), :]
                Y = Y[~jnp.isnan(X_obs)]
                X_obs = X_obs[~jnp.isnan(X_obs)]

            if normed:
                X_obs = X_obs/(2*jnp.pi)
            
            print("Timesteps x Neurons: " +str(Y.shape))
            return [Y, X_obs]
        
        elif epoch == "sleep":
            print("Timesteps x Neurons: " +str(Y.shape))
            return Y

        else:
            raise Exception(
            "Not a valid epoch"
        )


    def get_tuning(self):
        return self.tuning_curves

class GuillaumeLinearTrack:
    def __init__(self, filepath):
        data = nap.load_file(filepath)

        spikes = data["units"]
        position = data["position"]
        spikes = spikes.getby_category("cell_type")["pE"]
        print(spikes)
        self.trials = data["trials"]
        self.theta = data["theta_phase"]
        self.position = position.restrict(data["trials"])
        self.spikes = spikes.getby_threshold("rate", .7)
        self.num_trials = self.trials.shape[0]

    def load_data(self, bin_size, return_theta = True, normed = True, scale = 1.):
        
        #order = pf.idxmax().sort_values().index.values

        theta = self.theta.bin_average(bin_size, self.position.time_support)
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        trial_index = self.trials.in_interval(theta)


        if normed:
            self.position = ((self.position - 5)/165)*scale
        else:
            self.position = self.position*scale

        self.data = nap.TsdFrame(
            t=theta.t,
            d=np.vstack(
                (self.position.interpolate(theta, ep=self.position.time_support).values, theta.values, trial_index)
            ).T,
            time_support=self.position.time_support,
            columns=["position", "theta", "trial_index"]
        )

        Y = jnp.array(self.spikes.count(bin_size, ep=self.position.time_support))#[:20000,:])
        
        if return_theta:
            return [Y, jnp.array(self.data["position"].bin_average(bin_size, self.position.time_support))], self.data['theta']
        else:
            return [Y, jnp.array(self.data["position"].bin_average(bin_size, self.position.time_support))]

    def load_data_trials(self, bin_size, return_theta = True, normed = True, masked=False, scale = 1.):
        
        
        theta = self.theta.bin_average(bin_size, self.position.time_support)
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        trial_index = self.trials.in_interval(theta)

        if normed:
            self.position = ((self.position - 5)/165)*scale
        else:
            self.position = self.position*scale

        self.data = nap.TsdFrame(
            t=theta.t,
            d=np.vstack(
                (self.position.interpolate(theta, ep=self.position.time_support).values, theta.values, trial_index)
            ).T,
            time_support=self.position.time_support,
            columns=["position", "theta", "trial_index"]
        )
        
        if masked:
            S = [jnp.array(self.data["position"].bin_average(bin_size, self.trials[i])[:, None]) for i in range(self.num_trials)]
            
            self.s_shapes = jnp.array([i.shape[0] for i in S])
            longest_trial = jnp.max(self.s_shapes)
            num_neurons = jnp.array(self.spikes.count(bin_size, ep=self.trials[0])).shape[1]
            
            '''Y = {}
            for i in range(self.num_trials):
                S_mask = np.full((longest_trial, 1), np.nan)
                Y_mask = np.full((longest_trial, num_neurons), np.nan)
                
                S_mask[:S[i].shape[0],:]  = S[i]
                Y_mask[:S[i].shape[0],:] = self.spikes.count(bin_size, ep=self.trials[i])
   
                Y.update({i:tuple([jnp.array(Y_mask), jnp.array(S_mask)])})'''
            Y = []

            S_mask = np.zeros((self.num_trials, longest_trial, 1))#np.full((self.num_trials, longest_trial, 1), 0)#np.nan)
            Y_mask = np.zeros((self.num_trials, longest_trial, num_neurons))#np.full((self.num_trials, longest_trial, num_neurons), 0)#np.nan)
            theta_mask = np.zeros((self.num_trials, longest_trial))
            for i in range(self.num_trials):
                
                S_mask[i, :S[i].shape[0],:]  = S[i]
                Y_mask[i, :S[i].shape[0],:] = self.spikes.count(bin_size, ep=self.trials[i])
                theta_mask[i, :S[i].shape[0]] = np.array(self.data["theta"].bin_average(bin_size, self.trials[i]))
            
            Y = [tuple([jnp.array(Y_mask), jnp.array(S_mask)]), 
                    tuple([jnp.where(Y_mask==0, 0, 1),
                        jnp.where(S_mask==0, 0, 1)])]
            
        
        else:
            Y = {i:tuple([jnp.array(self.spikes.count(bin_size, ep=self.trials[i])),
                            np.array(self.data["position"].bin_average(bin_size, self.trials[i])[:, None])]) for i in range(self.num_trials)}
            
        if return_theta:
            return Y, theta_mask
        else:
            return Y
        

    def get_tuning(self, n_bins):
        pf = nap.compute_1d_tuning_curves(self.spikes, self.data["position"], n_bins, self.data["position"].time_support)
        return jnp.array(pf.T)

class BurakGridCells:
    def __init__(self, filepath, scale = 1.):
        
        dataset = np.load(filepath, allow_pickle=True).item()
        
        of_light = dataset['task'][1]
        of_dark = dataset['task'][0]
        light_spikes = dataset['task'][1]['spike_timestamp']
        dark_spikes = dataset['task'][0]['spike_timestamp']

        dark_timestamps = dataset['task'][0]['tracking']['t']
        light_timestamps = dataset['task'][1]['tracking']['t']

        t = np.concatenate((dataset['task'][0]['tracking']['t'],
                            dataset['task'][1]['tracking']['t']))
        x = ((np.concatenate((dataset['task'][0]['tracking']['x'],
                            dataset['task'][1]['tracking']['x']))))#+80)/150)*scale
        x = ((x - np.min(x))/150)*scale

        y = ((np.concatenate((dataset['task'][0]['tracking']['y'],
                            dataset['task'][1]['tracking']['y']))))#+80)/150)*scale
        y = ((y - np.min(y))/150)*scale

        
        z = np.concatenate((dataset['task'][0]['tracking']['z'],
                            dataset['task'][1]['tracking']['z']))
        hd = np.concatenate((dataset['task'][0]['tracking']['hd'],
                            dataset['task'][1]['tracking']['hd']))

        metadata = {"condition": ["dark", "light"]}
        self.epochs = nap.IntervalSet(start = [np.min(dark_timestamps), np.min(light_timestamps)],
                                end = [np.max(dark_timestamps), np.max(light_timestamps)], time_units = 's', metadata=metadata)
        neurons = {}
        mod = []
        for i in range(len(dataset['unit_id'])): 
            curr_id = dataset['unit_id'][i]
            
            spike_inds_dark = np.where(dataset['task'][0]['spike_cluster_id'] == curr_id)[0]
            spikes_dark = dataset['task'][0]['spike_timestamp'][spike_inds_dark]
            spike_inds_light = np.where(dataset['task'][1]['spike_cluster_id'] == curr_id)[0]
            spikes_light = dataset['task'][1]['spike_timestamp'][spike_inds_light]
            spikes = np.concatenate((spikes_dark, spikes_light))

            neurons.update({curr_id:nap.Ts(t=spikes, time_units='s', time_support=self.epochs)})
            mod.append('mod'+str(int(dataset['module_id'][i])))
        
        self.spikes = nap.TsGroup(neurons, module=np.array(mod))

        self.position = nap.TsdFrame(t=t, d=np.stack((x, y, z, hd)).T, columns=['x','y','z','hd'], time_support=self.epochs)


    def load_data_by_module(self, module, epoch, bin_size, n_bins = 20):
        spiking = self.spikes.getby_category("module")[module]#'mod1']

        self.tc, self.binsxy = nap.compute_2d_tuning_curves(spiking, 
                self.position['x','y'].restrict(self.epochs[self.epochs.condition == epoch]), n_bins)#self.epochs[self.epoch_keys[epoch]]), n_bins)
        for i in self.tc.keys():
            self.tc[i] = self.tc[i]*bin_size
        Y = jnp.array(spiking.count(bin_size, ep=self.epochs[self.epochs.condition == epoch]))#self.epochs[self.epochs == epoch]))
        

        X_obs = jnp.vstack((jnp.array(self.position['x'].bin_average(bin_size, ep=self.epochs[self.epochs.condition == epoch])),#self.epochs[self.epoch_keys[epoch]])),
                        jnp.array(self.position['y'].bin_average(bin_size, ep=self.epochs[self.epochs.condition == epoch])))).T#self.epochs[self.epoch_keys[epoch]])))).T
        
        #print("Timesteps x Neurons: " +str(Y.shape))
        
        return [Y, X_obs]

    def load_data_by_index(self, inds, epoch, bin_size, n_bins = 20, ret_headdir=False):

        spiking = self.spikes[inds]
        self.tc, self.binsxy = nap.compute_2d_tuning_curves(spiking, 
                self.position['x','y'].restrict(self.epochs[self.epochs.condition == epoch]), n_bins)
        for i in self.tc.keys():
            self.tc[i] = self.tc[i]*bin_size
        Y = jnp.array(spiking.count(bin_size, ep=self.epochs[self.epochs.condition == epoch]))
        
        X_obs = jnp.vstack((jnp.array(self.position['x'].bin_average(bin_size, ep=self.epochs[self.epochs.condition == epoch])),
                        jnp.array(self.position['y'].bin_average(bin_size,  ep=self.epochs[self.epochs.condition == epoch])))).T
        
        #print("Timesteps x Neurons: " +str(Y.shape))
        if ret_headdir:
            headdir = jnp.array(self.position['hd'].bin_average(bin_size, ep=self.epochs[self.epochs.condition == epoch]))
                        
            return [Y, X_obs], headdir
        else:
            return [Y, X_obs]

    



    def get_tuning(self, return_bins = False):
        if return_bins:
            return self.tc, self.binsxy
        else:
            return self.tc

class MovshonV1:
    def __init__(self,filepath):
        self.data = loadmat(filepath,simplify_cells=True)

    def load_data(self, lower_bound = 140, upper_bound = 1140, normed = True,  data_format = "360"):
        '''' data formats = "360" for standard, "180_merged", "0_180", "180_360"'''
        spike_data = self.data['spk_times']
        time = self.data['t_sync']
        orientation = self.data['ori']
        print("TODO!!1 rescale by time if not 1 sec")
        if data_format=="180_merged":
            orientation = jnp.array([ori - 180 if ori >= 180 else ori for ori in orientation])
            sort_inds = jnp.argsort(orientation)
            self.n_conds = 36
            self.n_trials = 100
            if normed:
                orientation = orientation/180
        elif data_format=="0_180":
            
            sort_inds = jnp.argsort(orientation)
            sort_inds = sort_inds[jnp.where(orientation[sort_inds]<180)[0]]
            
            self.n_conds = 36
            self.n_trials = 50
            if normed:
                orientation = orientation/180
        elif data_format=="180_360":
            sort_inds = jnp.argsort(orientation)
            sort_inds = sort_inds[jnp.where(orientation[sort_inds]>=180)[0]]
            
            self.n_conds = 36
            self.n_trials = 50
            if normed:
                orientation = (orientation-180)/180
        elif data_format=="360":
            sort_inds = jnp.argsort(orientation)
            self.n_conds = 72
            self.n_trials = 50
            if normed:
                orientation = orientation/360

        self.orientation = orientation[sort_inds]
        self.original_inds = jnp.arange(len(self.orientation))[sort_inds]
        spikes_sorted = spike_data[:, sort_inds]
        #print(spikes_sorted.shape)
        n_neurons = spikes_sorted.shape[0]
        spikes_counts = np.zeros((n_neurons, self.n_conds*self.n_trials))
        for i in range(n_neurons):
            for j in range(self.n_conds*self.n_trials):
                trial = spikes_sorted[i,j]
                if isinstance(trial, int) or isinstance(trial, float):
                    if (lower_bound<trial) & (trial<upper_bound):
                        spikes_counts[i, j] = 1
                else:
                    filtered = np.array(trial[np.where((lower_bound<trial) &(trial<upper_bound))])
                    spikes_counts[i, j] = len(filtered)
        big_inds = np.argsort(-np.max(spikes_counts, axis=1))

        self.spikes_counts = spikes_counts[:,:]
        self.tuning_mean = np.mean(self.spikes_counts.reshape(n_neurons, self.n_conds, self.n_trials), axis=2)
        #print("Neurons x Trials: " + str(spikes_counts.shape)) 

        return [self.spikes_counts.T, self.orientation]

    def get_tuning(self, return_bins = False):
        return self.tuning_mean


class ShenoyMotor:
    def __init__(self,filepath):
        self.stimuli = np.load(filepath+"/1ring_x.npy")
        self.spikes = np.load(filepath+"/1ring_y.npy")

    def load_data(self, n_repeats = 60, normed = True, incl_m1 = False):
        if incl_m1:
            spikes = self.spikes[:n_repeats, :, :]
        else:    
            spikes = self.spikes[:n_repeats, :, 96:]
        #spikes = spikes - np.mean(np.mean(spikes, axis=0), axis = 0)
        n_trials, n_conditions, n_neurons = spikes.shape

        tuning_curves = jnp.mean(spikes, axis = 0)

        pref_ang_sorted = jnp.argsort(jnp.argmax(tuning_curves, axis = 0))

        tuning = tuning_curves[:, pref_ang_sorted]
        spikes = spikes[:,:,pref_ang_sorted].reshape(n_trials*n_conditions, n_neurons)

        self.stimuli = np.tile(np.linspace(0, 345, 24), 60).flatten()
        if normed:
            self.stimuli = self.stimuli/360

        self.spikes = spikes
        self.tuning = tuning.T
        print("Neurons x Trials: " + str(spikes.shape)) 

        return [self.spikes, self.stimuli]

    def get_tuning(self, return_bins = False):
        return self.tuning
