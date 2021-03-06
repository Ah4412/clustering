from numpy import *
from numpy.random import randint
import hashlib
from six import iteritems
#from six.moves import range
from logger import log_message
#from hamming_maskstarts import hamming_maskstarts
from compute_penalty import compute_penalty
from m_step import compute_cluster_bern
from log import comp_log
#from e_step import compute_cluster_subresponsibility, compute_log_p_and_assign
from e_step import compute_subresponsibility, compute_log_p_and_assign
# compute_cluster_bern
from default_parameters import default_parameters
import time
import pyopencl as cl
from IPython import embed

class PartitionError(Exception):
    pass


class section(object):
    def __init__(self, kk, name, *args, **kwds):
        self.kk = kk
        self.name = name
        self.args = args
        self.kwds = kwds
        if not hasattr(kk, '_section_timings_t_total'):
            kk._section_timings_t_total = {}
            kk._section_timings_num_calls = {}
        if name not in kk._section_timings_t_total:
            kk._section_timings_t_total[name] = 0.0
            kk._section_timings_num_calls[name] = 0
    def __enter__(self):
        self.kk.run_callbacks('start_'+self.name, *self.args, **self.kwds)
        self.t_start = time.time()
    def __exit__(self, type, value, traceback):
        this_time = time.time()-self.t_start
        self.kk.run_callbacks('end_'+self.name, *self.args, **self.kwds)
        if self.kk.name:
            self.kk.log('debug', 'This call: %.2f ms.' % (this_time*1000),
                        suffix='timing.'+self.name)
            return
        name = self.name
        st_t = self.kk._section_timings_t_total
        st_n = self.kk._section_timings_num_calls
        st_t[name] += this_time
        st_n[name] += 1
        mean_time = st_t[name]/st_n[name]
        self.kk.log('debug', 'This call: %.2f ms. Average: %.2f ms. Total: %.2f s. '
                             'Num calls: %d' % (this_time*1000, mean_time*1000,
                                                st_t[name], st_n[name]),
                    suffix='timing.'+self.name)


def add_slots(meth):
    def new_meth(self, *args, **kwds):
        with section(self, meth.__name__, *args, **kwds):
            res = meth(self, *args, **kwds)
        return res
    new_meth.__name__ = meth.__name__
    new_meth.__doc__ = meth.__doc__
    return new_meth

class KK(object):
    '''
    Main object used for clustering the supercluster points
    * data = supercluster data
    * Initialisation KK(data,  **params)
    * Method kk.cluster_mask_starts(num_starting_clusters) will cluster from mask starts - 
      NOT YET FUNCTIONAL
    * Method kk.cluster_from(clusters) will cluster from the given array of cluster assignments.
      -USING THIS METHOD INSTEAD
    * kk.clusters (after clustering) is the array of cluster assignments.
    * Method kk.register_callback(callback, slot=None) register a callback function that will be
      called at the given slot (see code for slot names), by default at the end of each iteration
      of the algorithm. Callback functions are normally called as f(kk), but some slots will
      defined additional arguments and keyword arguments.
    '''
    def __init__(self, data, callbacks=None, name = '', is_subset = False, 
                 is_copy=False, map_log_to_debug=False, **params):
        
        self.name = name
        if callbacks is None:
            callbacks = {}
        self.callbacks = callbacks
        self.data = data
        self.cluster_hashes = set()
        self.is_subset = is_subset
        self.is_copy = is_copy
        self.map_log_to_debug = map_log_to_debug
        # user parameters
        show_params = name=='' and not is_subset  and not is_copy
        self.params = params
        actual_params = default_parameters.copy()
        for k, v in iteritems(params):
            if k not in default_parameters:
                raise ValueError("There is no parameter "+k)
            actual_params[k] = v
        for k, v in iteritems(actual_params):
            setattr(self, k, v)
            if show_params:
                self.log('info', '%s = %s' % (k, v), suffix='initial_parameters')
        self.all_params = actual_params #dictionary of parameters
        
    def register_callback(self, callback, slot='end_iteration'):
        if slot not in self.callbacks:
            self.callbacks[slot] = []
        self.callbacks[slot].append(callback) # callback dictionary

    def run_callbacks(self, slot, *args, **kwds):
        if slot in self.callbacks:
            for callback in self.callbacks[slot]:
                callback(self, *args, **kwds)

    def log(self, level, msg, suffix=None):
        if self.map_log_to_debug:
            level = 'debug'
        if suffix is not None:
            if self.name=='':
                name = suffix
            else:
                name = self.name+'.'+suffix
        else:
            name = self.name
        log_message(level, msg, name=name)        

    def copy(self, name='kk_copy',
             **additional_params):
        if self.name:
            sep = '.'
        else:
            sep = ''
        params = self.params.copy()
        params.update(**additional_params)
        return KK(self.data, name=self.name+sep+name,
                  callbacks=self.callbacks,
                  is_copy=True,
                  **params)

    def subset(self, spikes, name='kk_subset', **additional_params):
        newdata = self.data.subset(spikes)
        if self.name:
            sep = '.'
        else:
            sep = ''
        params = self.params.copy()
        params.update(**additional_params)
        return KK(newdata, name=self.name+sep+name,
                  callbacks=self.callbacks,
                  is_subset=True,
                  **params)      
      
 
    def initialise_clusters(self, clusters):
        self.clusters = clusters
        self.old_clusters = -1*ones(len(self.clusters), dtype=int)
        self.reindex_clusters()  

    def cluster_hammingmask_starts(self,):
        '''Start from hamming sorted set of clusters'''
        clusters = hammingmask_starts(self.data, self.num_starting_clusters)
        #clump_fine_clustering
        self.cluster_from(clusters)

    def cluster_from(self, clusters, recurse=True, score_target=-inf):
        self.log('info', 'Clustering data set of %d points, %d KKruns' % (self.data.num_spikes,
                                                                            self.data.num_KKruns))
        self.initialise_clusters(clusters)
        return self.iterate(recurse=recurse, score_target=score_target)
    
    def prepare_for_iterate(self):
        self.current_iteration = 0
        self.score_history = []
        self.cluster_distribution_history = []
        if self.save_prelogresponsibility:
            self.prelogresponsibility_history = []
    
    def iterate(self, recurse=True, score_target=-inf):        
        self.prepare_for_iterate()

        score = score_raw = score_penalty = None

        iterations_until_next_split = self.split_first
        tried_splitting_to_escape_cycle_hashes = set()

        self.log('info', 'Starting iteration 0 with %d clusters' % self.num_clusters_alive)

        while self.current_iteration<self.max_iterations:
            self.log('debug', 'Starting iteration %d' % self.current_iteration)
            self.MEC_steps()
            #embed()
            self.compute_penalty() 
            estep_score, estep_score_raw, estep_score_penalty = self.compute_score()
            self.score_history.append((estep_score, estep_score_raw, estep_score_penalty, 'pure_e_step'))#,self.num_cluster_members))
         #   print('score_history ', self.score_history)
            self.cluster_distribution_history.append((self.num_cluster_members, 'pure_e_step'))
            #embed()
            if recurse and self.consider_cluster_deletion and self.num_clusters_alive>2:
                #embed()
         #       print('recurse = ', recurse)
                self.consider_deletion()
            old_score = score
            old_score_raw = score_raw
            old_score_penalty = score_penalty
         #   print('pre_compute_score',score, score_raw, score_penalty)
            score, score_raw, score_penalty = self.compute_score()
            #self.score_history.append((score, score_raw, score_penalty, 'post_deletion'))#,self.num_cluster_members))
            #print('score_history ', self.score_history)
            #self.cluster_distribution_history.append((self.num_cluster_members,'post_deletion'))
            
            clusters_changed, = (self.clusters!=self.old_clusters).nonzero()
            clusters_changed = array(clusters_changed, dtype=int)
            num_changed = len(clusters_changed)
           # if num_changed:
          #      # add these changed clusters to all the candidate sets
           #     num_candidates = 0
           #     max_candidates = min(self.max_candidates,
            #        self.max_candidates_fraction*self.num_spikes*self.num_clusters_alive)
           #     with section(self, 'union'):
            #        for cluster, candidates in list(self.candidates.items()):
                        #candidates = union1d(candidates, clusters_changed)
                        #self.candidates[cluster] = candidates
                        #num_candidates += len(candidates)
                        #if num_candidates>max_candidates:
                            #self.candidates = dict()
                            ##self.force_next_step_full = True
                            #if num_candidates>self.max_candidates:
                                #self.log('info', 'Ran out of storage space, try increasing '
                                                 #'max_candidates if this happens often.')
                            #else:
                                #self.log('debug', 'Exceeded quick step point fraction, next step '
                                                  #'will be full')
                            #break

            self.run_callbacks('scores', score=score, score_raw=score_raw,
                               score_penalty=score_penalty, old_score=old_score,
                               old_score_raw=old_score_raw, old_score_penalty=old_score_penalty,
                               num_changed=num_changed,
                               )

            self.current_iteration += 1

            msg = 'Iteration %d: %d clusters, %d changed, score=%f' % (self.current_iteration,
                                                                         self.num_clusters_alive,
                                                                         num_changed, score)

            #last_step_full = self.full_step
            #self.full_step = (num_changed>self.num_changed_threshold*self.num_spikes or
            #                  num_changed==0 or
            #                  self.current_iteration % self.full_step_every == 0 or
            #                  (old_score is not None and score > old_score))
           # if not hasattr(self, 'old_log_p_best'):
            #    self.full_step = True 
            # We are no longer concerned about whether or not steps are full

            self.reindex_clusters()
            if old_score is not None:
                msg += ' (decreased by %f)' % (old_score-score)
            self.log('info', msg)
            if old_score is not None:
                msg = 'Change in scores: raw=%f, penalty=%f, total=%f'  % (old_score_raw-score_raw,
                                                                           old_score_penalty-score_penalty,
                                                                           old_score-score)
                print(msg)
                self.log('debug', msg)
            if (old_score is not None) and old_score-score <0:
                print('WARNING: The score has gone up, this should never happen \n Try to debug it')
                print('score =', score, ' > old_score = ', old_score)
                self.log('warning', 'WARNING: The score has gone up, this should never happen \n Try to debug it')
                self.score_history.append('score increase!')
                self.cluster_distribution_history.append('score increase!')
                if self.embed:
                    embed()    

            # Splitting logic
            iterations_until_next_split -= 1
            if num_changed==0:
                self.log('info', 'No points changed, so trying to split.')
                iterations_until_next_split = 0

            # Cycle detection/breaking
            cluster_hash = hashlib.sha1(self.clusters.view(uint8)).hexdigest()
            if cluster_hash in self.cluster_hashes and num_changed>0:
                if recurse:
                    if cluster_hash in tried_splitting_to_escape_cycle_hashes:
                        self.log('error', 'Cycle detected! Already tried attempting to break out '
                                          'by splitting, so abandoning.')
                        break
                    else:
                        self.log('warning', 'Cycle detected! Attempting to break out by splitting.')
                        iterations_until_next_split = 0
                    tried_splitting_to_escape_cycle_hashes.add(cluster_hash)
                else:
                    self.log('error', 'Cycle detected! Splitting is not enabled, so abandoning.')
                    break
            self.cluster_hashes.add(cluster_hash)

            # Try splitting
            did_split = False
            if recurse and iterations_until_next_split<=self.min_points_split_cluster:
                
                did_split = self.try_splits()
                iterations_until_next_split = self.split_every

            self.run_callbacks('end_iteration')

            if num_changed==0 and not did_split:
                self.log('info', 'No points changed, and did not split, '
                                 'so finishing.')
                break

            if num_changed<self.break_fraction*self.num_spikes:
                self.log('info', 'Number of points changed below break fraction, so finishing.')
                break

            if score<score_target:
                self.log('info', 'Reached score target, so finishing.')
        else:
            # ran out of iterations
            self.log('info', 'Number of iterations exceeded maximum %d' % self.max_iterations)

        return score
    
    @add_slots
    def MEC_steps(self, only_evaluate_current_clusters=False):
        # eliminate any clusters with 0 members, compute the list of spikes
        # in each cluster, compute the weights and generalized Bernoulli
        #parameters 
        self.reindex_clusters()
        # Computes the masked and unmasked indices for each cluster based on the
        # masks for each point in that cluster. Allocates space for covariance
        # matrices.
        print('MEC_steps:\n')
        num_clusters = self.num_clusters_alive
        num_KKruns = self.num_KKruns
        num_cluster_members = self.num_cluster_members
        cluster_start = 0
        num_spikes = self.num_spikes
        max_Dk = amax(self.D_k)
        max_Dk_size = max_Dk + 1

        # Weight computations \pi_c
        denom = self.num_spikes
        denom = float(denom)
        
        # Arrays that will be used in E-step part
        if only_evaluate_current_clusters:
            self.clusters_second_best = zeros(0, dtype=int)
            self.log_p_best = -inf*ones(num_spikes)
           # self.log_p_best = empty(num_spikes)
            self.log_p_second_best = empty(0)
        else:    
            self.old_clusters = self.clusters
            self.clusters = -ones(num_spikes, dtype=int) #set them to -1 to avoid bugs
            self.clusters_second_best = -ones(num_spikes, dtype=int)
            if hasattr(self, 'log_p_best'):
                self.old_log_p_best = self.log_p_best
            self.log_p_best = -inf*ones(num_spikes)
            self.log_p_second_best = -inf*ones(num_spikes)
        

        num_skipped = 0
        
       # if not only_evaluate_current_clusters:
       #     self.log_p_best[:] = 0
        
      #  if only_evaluate_current_clusters:
      #      self.candidates = dict() # replaces quick_step_candidates
      #      for cluster in range(num_clusters):
      #          self.candidates[cluster] = self.get_spikes_in_cluster(cluster)
      #      self.collect_candidates = False
      #  else:
      #      self.candidates = dict()
      #      self.collect_candidates = True
        
        
        clusters_to_kill = []
        
        #bern = zeros((num_clusters, num_KKruns, max_Dk_size), dtype = float)
        num_bern_params = zeros(num_clusters, dtype = int)
        log_bern2 = zeros((num_clusters, num_KKruns, max_Dk_size), dtype = float32) 
        log_bern = zeros((num_clusters, num_KKruns, max_Dk_size), dtype = float)

        debug = zeros((num_clusters,num_spikes), dtype = float32)

        platforms = cl.get_platforms()
        if len(platforms) == 0:
            quit()
        # for i in range(0, len(platforms)):
        #     print(i, " : ", platforms[i])
        selected_platform = 0#input("Select the desired platform: ")
        platform = platforms[int(selected_platform)]

        devices = platform.get_devices(device_type=cl.device_type.ALL)
        if len(devices) == 0:
            quit()
        # for i in range(0, len(devices)):
        #     print(i, " : ", devices[0])
        selected_device = 0#input("Select the desired devices: ")
        device = devices[int(selected_device)]

        with open("Kernels/kernel_g.cl") as myfile:
            source = myfile.read()
       # prelogresponsibility = zeros((num_clusters, num_spikes), dtype = float)
        #preresponsibility = zeros((num_clusters, num_spikes), dtype = float)
        ########### M step ########################################################
        # Normalize by total number of points to give class weight

        ctx = cl.Context([device])
        prg = cl.Program(ctx, source).build()
        queue = cl.CommandQueue(ctx)    

        mf = cl.mem_flags

        nspikes_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(self.num_spikes))
        nKKruns = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(num_KKruns))
        nmaxDk = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(max_Dk_size))
        sparsekks_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(self.data.supersparsekks[:,0]))
        sparsecls_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(self.data.supersparsekks[:,1]))
        start_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(self.data.super_start))
        end_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(self.data.super_end))
        bern_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=float32(log_bern2))
        spikes_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(self.spikes_in_cluster))
        s_offset_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=int32(self.spikes_in_cluster_offset))
        debug1 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=float32(debug))


        prg.mstep(queue, log_bern.shape, None, nKKruns, nmaxDk, sparsekks_g, sparsecls_g, start_g, end_g, spikes_g, s_offset_g, bern_g)

        cl.enqueue_copy(queue, log_bern2, bern_g)
        
        # embed()

        y = log_bern2.sum(axis = 2)
        x = diff(self.spikes_in_cluster_offset)

        for i in range(len(x)):
            log_bern2[i,:,0] = x[i]   

        log_bern2[:,:,0] = log_bern2[:,:,0] - y[:,:]

        for cluster in range(num_clusters): 
            [log_cluster_bern, num_bern_params[cluster]] = compute_cluster_bern(self, cluster, max_Dk) 
            log_bern[cluster,:,:] = log_cluster_bern             


        log_bern2 = comp_log(log_bern2)


        prelogresponsibility2 = zeros((num_clusters, num_spikes), dtype = float32)
        filler = zeros(num_clusters, dtype = float32)

        weights = (num_cluster_members)/denom
        for cluster in range(num_clusters):
            all_zero_sum = sum(log_bern2[cluster, isfinite(log_bern[cluster,:,0]), 0])
            filler[cluster] = log(weights[cluster])- num_KKruns*log(x[cluster]) + all_zero_sum
            
        prelog_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=float32(prelogresponsibility2))
        filler_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=float32(filler))
        bern_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=float32(log_bern2))


        prg.prelog(queue, prelogresponsibility2.shape, None, nKKruns, nmaxDk, nspikes_g, sparsekks_g, sparsecls_g, start_g, end_g, bern_g, prelog_g, filler_g, debug1)

        cl.enqueue_copy(queue, debug, debug1)
        cl.enqueue_copy(queue, prelogresponsibility2, prelog_g)


        prelogresponsibility =  compute_subresponsibility(self, weights,  log_bern, num_clusters) 
            #unbern[cluster,:,:]=bern[cluster,:,:]*len(self.get_spikes_in_cluster(cluster))
        

        nclus_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=int32(prelogresponsibility2.shape[0]))
        lpb_g  = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=float32(self.log_p_best))
        lpsb_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=float32(self.log_p_second_best))
        clus_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=int32(self.clusters))
        sclus_g= cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=int32(self.clusters_second_best))
        ooec_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=int32(only_evaluate_current_clusters))


        t_log_p_best = zeros(self.log_p_best.shape, dtype = float32)
        t_log_p_second_best = zeros(self.log_p_second_best.shape, dtype = float32)
        t_clusters = zeros(self.clusters.shape, dtype = int32)
        t_clusters_second_best = zeros(self.clusters_second_best.shape, dtype = int32)
        
        prg.assign(queue, self.clusters.shape, None, nspikes_g, prelog_g, lpb_g, lpsb_g, clus_g, sclus_g, nclus_g,ooec_g, debug1)

        cl.enqueue_copy(queue, debug, debug1)
        cl.enqueue_copy(queue, t_log_p_best, lpb_g)
        cl.enqueue_copy(queue, t_log_p_second_best, lpsb_g)
        cl.enqueue_copy(queue, t_clusters, clus_g)
        cl.enqueue_copy(queue, t_clusters_second_best, sclus_g)

        


        if self.embed:    
            self.log_bern = log_bern    
            self.prelogresponsibility = prelogresponsibility
        if self.save_prelogresponsibility:
            self.prelogresponsibility_history.append(self.prelogresponsibility)
        self.num_bern_params = num_bern_params
        #sumresponsibility = sum(preresponsibility, axis = 0)
        #responsibility = preresponsibility/sumresponsibility
        self.run_callbacks('e_step_before_main_loop',  cluster=cluster,
                          )
       # if only_evaluate_current_clusters:
       #     embed()
        compute_log_p_and_assign(self, prelogresponsibility, only_evaluate_current_clusters)       
        #compute_log_p_and_assign(self, weights, bern, only_evaluate_current_clusters)
            
        self.run_callbacks('e_step_after_main_loop')
        #embed()
        # we've reassigned clusters so we need to recompute the partitions, but we don't want to
        # reindex yet because we may reassign points to different clusters and we need the original
        # cluster numbers for that

        a =t_log_p_best-self.log_p_best
        b =t_log_p_second_best - self.log_p_second_best
        c = t_clusters-self.clusters
        d = t_clusters_second_best - self.clusters_second_best
        embed()
        


        self.partition_clusters()

    @add_slots
    def compute_penalty(self, clusters=None):
        penalty = compute_penalty(self, clusters)
        if clusters is None:
            self.penalty = penalty
        return penalty

    @add_slots
    def consider_deletion(self):
        print('attempting deletion \n')
        num_cluster_members = self.num_cluster_members
        num_clusters = self.num_clusters_alive
        
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        log_p_best = self.log_p_best
        log_p_second_best = self.log_p_second_best

        deletion_loss = zeros(num_clusters)
        I = arange(self.num_spikes)
        #add.at(deletion_loss, self.clusters, log_p_second_best-log_p_best)
        #add.at(deletion_loss, self.clusters, log_p_best-log_p_second_best)
        add.at(deletion_loss, self.clusters, 2*(log_p_best-log_p_second_best))
        #embed()
        score, score_raw, score_penalty = self.compute_score()
        #self.score_history.append((score, score_raw, score_penalty, 'pre_deletion'))#,self.num_cluster_members))
        #self.cluster_distribution_history.append((self.num_cluster_members,'pre_deletion'))
        candidate_cluster = -1
        improvement = -inf
        #embed()
        #We  only delete a single cluster at a time, 
        #so we pick the optimal candidate for deletion
        for cluster in range(num_clusters):
            new_clusters = self.clusters.copy()
            # reassign points
            cursic = sic[sico[cluster]:sico[cluster+1]]
            new_clusters[cursic] = self.clusters_second_best[cursic]
            # compute penalties if we reassigned this
           # embed()
            new_penalty = self.compute_penalty(new_clusters)
            new_score = score_raw + deletion_loss[cluster] + new_penalty
           # print('SCORE_RAW', score_raw)
           # print('deletion_loss[%g] ='%cluster, deletion_loss[cluster])
           # print('new score =', new_score)
           # print('new_penalty = ', new_penalty)
            cur_improvement = score-new_score # we want improvement to be a positive value
            #embed()
            if cur_improvement>improvement:
                improvement = cur_improvement
                candidate_cluster = cluster
          #  print('candidate_cluster ',    candidate_cluster)  
        #embed()
        if improvement>0:
            # delete this cluster
           # print('WE ARE DELETING A CLUSTER')
            num_points_in_candidate = sico[candidate_cluster+1]-sico[candidate_cluster]
            #self.log('info', 'Deleting cluster {cluster} ({numpoints} points): improves score '
            #                 'by {improvement}'.format(cluster=candidate_cluster,
            #                                           numpoints=num_points_in_candidate,
            #                                           improvement=improvement))
            # reassign points
            cursic = sic[sico[candidate_cluster]:sico[candidate_cluster+1]]
            clustersk4 = self.clusters.copy()
            clustersk4[cursic] = self.clusters_second_best[cursic]
            
            with section(self, 'deletion_evaluation'):
                # will deletion really improve the score as the M-Step determined variables have now changed?
             #   print('evaluation of deletion with K4')
                K4 = self.copy(name='deletion_evaluation', map_log_to_debug=True)
                #clusters = self.clusters.copy()
                K4.initialise_clusters(clustersk4)
                K4.prepare_for_iterate()
                K4.MEC_steps(only_evaluate_current_clusters=True)
                K4.compute_penalty()
                score_aftermstepdel, raw_aftermstepdel, penalty_aftermstepdel  = K4.compute_score()
              #  print('afterscore_del = ',score_aftermstepdel,raw_aftermstepdel,penalty_aftermstepdel)
            #embed()
            if score_aftermstepdel>score:
                print('NO DELETION WILL OCCUR')
                #self.clusters = self.old_clusters
                self.score_history.append('score increase successfully averted, even though suggested improvement was %d!'%(improvement))
                self.cluster_distribution_history.append('score increase successfully averted, even though suggested improvement was %d!'%(improvement))
            else: 
                print('WE ARE DELETING A CLUSTER')
                self.clusters = K4.clusters.copy()
                self.reindex_clusters()
                self.log('info', 'Deleting cluster {cluster} ({numpoints} points): improves score '
                             'by {improvement}'.format(cluster=candidate_cluster,
                                                       numpoints=num_points_in_candidate,
                                                       improvement=improvement))
                print('debug', 'Score improved after deleting cluster '
                                  '%d ' % (candidate_cluster))
                self.log('debug', 'Score improved after deleting cluster '
                                  '%d ' % (candidate_cluster))                
            ##self.log_p_best[cursic] = self.log_p_second_best[cursic]
            ## at this point we have invalidated the partitions, so to make sure we don't miss
            ## something, we wipe them out here
            ##self.partition_clusters()
            ##self.compute_penalty() # and recompute the penalties
            ## we've also invalidated the second best log_p and clusters
            ##self.log_p_second_best = None
            ##self.clusters_second_best = None
            ## and we will need to do a full step next time
            ##self.force_next_step_full = True
                #self.compute_penalty()
                #postscore, postscore_raw, postscore_penalty = self.compute_score()
                self.score_history.append(('improvement', improvement))
                self.cluster_distribution_history.append(('improvement', improvement))
                self.score_history.append((score_aftermstepdel, raw_aftermstepdel, penalty_aftermstepdel,'K4'))
                #self.score_history.append((postscore, postscore_raw, postscore_penalty, 'post_deletionlit'))#,self.num_cluster_members))
                print('score_history ', self.score_history)
                self.cluster_distribution_history.append((K4.num_cluster_members,'K4'))
                #self.cluster_distribution_history.append((self.num_cluster_members,'post_deletionlit'))
                

    @add_slots
    def compute_score(self):
        #essential_params = self.num_clusters_alive*self.num_KKruns*(sum(self.D_k)-self.num_KKruns) #\sum_{k=1}^{num_KKruns} D(k)
        penalty = self.penalty
     #  print('PENALTY', penalty)
        raw = -2*sum(self.log_p_best) #Check this factor AIC = 2k-2log(L)
        #raw = 2*sum(self.log_p_best)
        score = raw+penalty
        self.log('debug', 'compute_score: raw %f + penalty %f = %f' % (raw, penalty, score))
        #print('debug', 'compute_score: raw %f + penalty %f = %f' % (raw, penalty, score))
        return score, raw, penalty
    
    @property
    def D_k(self):#'vector of the number of different clusters returned by each run of local KK'
        return self.data.D_k
    
    @property
    def num_spikes(self):
        return self.data.num_spikes

    @property
    def num_KKruns(self):
        return self.data.num_KKruns

    @property
    def num_clusters_alive(self):
        return len(self.num_cluster_members)

    @add_slots
    def reindex_clusters(self):
        '''
        Remove any clusters with 0 members (except for clusters 0 and 1),
        and recompute the list of spikes in each cluster. After this function is
        run, you can use the attributes:

        - num_cluster_members (of length the number of clusters)
        - spikes_in_cluster, spikes_in_cluster_offset

        spikes_in_cluster[spikes_in_cluster_offset[c]:spikes_in_cluster_offset[c+1]] will be in the indices
        of all the spikes in cluster c.
        '''
        num_cluster_members = array(bincount(self.clusters), dtype=int)
        I = num_cluster_members>0
        #I[0:self.num_special_clusters] = True # we keep special clusters
        remapping = hstack((0, cumsum(I)))[:-1]
        self.clusters = remapping[self.clusters]
        total_clusters = sum(I)
        if hasattr(self, '_total_clusters') and total_clusters<self._total_clusters:
            #self.force_next_step_full = True
            if hasattr(self, 'clusters_second_best'):
                del self.clusters_second_best
        self._total_clusters = total_clusters
        self.partition_clusters()

    def partition_clusters(self, clusters=None):
        if clusters is None:
            clusters = self.clusters
            assign_to_self = True
        else:
            assign_to_self = False
        try:
            num_cluster_members = array(bincount(clusters), dtype=int)
        except ValueError:
            #print(clusters)
            raise PartitionError
        I = array(argsort(clusters), dtype=int)
        y = clusters[I]
        n = amax(y)
      #  if n<self.num_special_clusters-1:
      #      n = self.num_special_clusters-1
        n += 2
        J = searchsorted(y, arange(n))
        if assign_to_self:
            self.num_cluster_members = num_cluster_members
            self.spikes_in_cluster = I
            self.spikes_in_cluster_offset = J
        else:
            return I, J, num_cluster_members

    def invalidate_partitions(self):
        self.num_cluster_members = None
        self.spikes_in_cluster = None
        self.spikes_in_cluster_offset = None

    def get_spikes_in_cluster(self, cluster):
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        return sic[sico[cluster]:sico[cluster+1]]
        
    @add_slots
    def try_splits(self):
        did_split = False
        num_clusters = self.num_clusters_alive

        self.log('info', 'Trying to split clusters')

        score_ref = None

        self.reindex_clusters()

        for cluster in range(num_clusters):
            if num_clusters>=self.max_possible_clusters:
                self.log('info', 'No more splitting, already at maximum number of '
                                 'clusters: %d' % self.max_possible_clusters)
                return did_split

            spikes_in_cluster = self.get_spikes_in_cluster(cluster)
            #if len(spikes_in_cluster)==0:
            if len(spikes_in_cluster)<=5:
                continue

            with section(self, 'split_candidate'):
                if self.max_split_iterations is not None:
                    max_iter = self.max_split_iterations
                else:
                    max_iter = self.max_iterations

                K2 = self.subset(spikes_in_cluster, name='split_candidate',
                                 max_iterations=max_iter,
                                 map_log_to_debug=True,
                                 )
                # at this point in C++ code we look for an unused cluster, but here we can just
                # use num_clusters+1
                self.log('debug', 'Trying to split cluster %d containing '
                                  '%d points' % (cluster, len(spikes_in_cluster)))
                # initialise with current clusters, do not allow creation of new clusters
                K2.max_possible_clusters = 1
                clusters = full(len(spikes_in_cluster), 0, dtype=int)
                try:
                    unsplit_score = K2.cluster_from(clusters, recurse=False)
                except PartitionError:
                    self.log('error', 'Partitioning error on split, K2.clusters = %s' % K2.clusters)
                    continue
                self.run_callbacks('split_k2_1', cluster=cluster, K2=K2,
                                   unsplit_score=unsplit_score)
                # initialise randomly, allow for one additional cluster
                K2.max_possible_clusters = 2
                clusters = randint(0, 2, size=len(spikes_in_cluster))
                if amax(clusters)!=1:
                    continue

                #if self.fast_split:
                    #print('FAST SPLIT: score_target = ', score_target)
                    #score_target = unsplit_score
                #else:
               # print('score_target = will be -inf')
                score_target = -inf

                try:
                    split_score = K2.cluster_from(clusters, recurse=False,
                                                  score_target=score_target)
                except PartitionError:
                    self.log('error', 'Partitioning error on split, K2.clusters = %s' % K2.clusters)
                    continue
                self.run_callbacks('split_k2_2', cluster=cluster, K2=K2, split_score=split_score,
                                   unsplit_score=unsplit_score)

                if K2.num_clusters_alive==0:
                    self.log('error', 'No clusters alive in K2')
                    continue

                if split_score>=unsplit_score:
                    self.log('debug', 'Score after (%f) splitting worse than before (%f), '
                                      'so not splitting' % (split_score, unsplit_score))
                    continue

            

            with section(self, 'split_evaluation'):
                # will splitting improve the score in the whole data set?
                print('evaluation of split with K3')
                K3 = self.copy(name='split_evaluation', map_log_to_debug=True)
                clusters = self.clusters.copy()

                if score_ref is None:
                    K3.initialise_clusters(clusters)
                    K3.prepare_for_iterate()
                    K3.MEC_steps(only_evaluate_current_clusters=True)
                    K3.compute_penalty()
                    score_ref, _, _ = K3.compute_score()
                   # embed()

                #embed()
                I1 = (K2.clusters==1)
                clusters[spikes_in_cluster[I1]] = num_clusters # next available cluster
                
                K3.initialise_clusters(clusters)
                K3.prepare_for_iterate()
                K3.MEC_steps(only_evaluate_current_clusters=True)
                K3.compute_penalty()
                score_new, _, _ = K3.compute_score()
            
          #  print('score_ref = ', score_ref)
          #  print('score_new = ', score_new)
            if score_new<score_ref:
                print('debug', 'Score improved after splitting, so splitting cluster '
                                  '%d into %d' % (cluster, num_clusters))
                self.log('debug', 'Score improved after splitting, so splitting cluster '
                                  '%d into %d' % (cluster, num_clusters))
                did_split = True
                self.clusters = K3.clusters.copy()
                self.reindex_clusters()
                num_clusters = self.num_clusters_alive
                score_ref = score_new
            else:
                self.log('debug', 'Score got worse after splitting')

        # if we split, should make the next step full
        if did_split:
            #self.force_next_step_full = True
            self.log('info', 'Split into %d clusters' % num_clusters)

        return did_split
