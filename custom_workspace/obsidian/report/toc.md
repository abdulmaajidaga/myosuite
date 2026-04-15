  Front matter (not numbered):                                          
    Synopsis
    Acknowledgements
    Statement of Authorship
    Nomenclature
   
  Chapter 1   Introduction                     
    1.1  Research Motivation
	    what motivated me to research on such a topic
    1.2  Project Objectives                    
	    what objectives did i complete in the course of this research
    1.3  Dataset Overview
		    why am i even taking about the dataset in chapter 1, shouldnt that be somewhere else
    1.4  Overview of Approach and Results 
		    i think its wrong to show the approach and results already                    
    1.5  Applications of the Work 
		    i think even this shouldnt be here, lets follow a more scientific approach

  Chapter 2   Background
  ok so my research has a few sections, one is what im targeting that is upper limb impairment, what am i using to target it, deep learning and a musculoskeletal simulator, so models, sim, so motion synthesis covers the model part of it, the names could be more discrete. ok so if i have task and dataset collection why am i also having that in intro, i have to remove it from there
  is there anything else background should cover? yes the entire literature review is what background should cover so keep that in mind.
    2.1  Post-Stroke Upper-Limb Impairment 
    2.2  Motion Synthesis for Rehabilitation
    2.3  Musculoskeletal Modelling                   
    2.4  Task and Dataset Selection                    

maybe a more compact heading
  Chapter 3   Dataset and Data Augmentation      
		for data, this is how i want i to go,  first talk about the the datasets available and why i went with mhh from ulimb and maybe i think this section should be in the literature review? it should definitely go in the literature review. after talking about the dataset, go into the detail, how many subjects, how many trials for one and then what did i filter out before any work, the left handed, the non dominant and the corrupted. After that the actual pre processing i did on the data and then talk about what the data scarcity problem and then we explore data augmentation strategies, giving them figs and stats on the augmentation and compaaring the 3 methods and that should be it.
    3.1  Motion Capture Recordings                  
    3.2  Preprocessing
    3.3  Feature Representation 
    3.4  Data Augmentation 
      3.4.1  SMOTE Interpolation 
      3.4.2  DTW Morphing        
      3.4.3  Linear Baseline
      3.4.4  Comparison of Augmentation Methods 

so now that i have 3 methods of augmentation, i need to find the best augmentation strategy and the best model config, so what do i do? how do i check which is the best, simply put i dont and i run experiments with all, and what about the models, i try toi backup every single step i did with literature and tell them that i did this fist thanks to this paper and this then and this and with the 3 augmentation strategies. we can go throug all the experiments we did nicely and come with a conclusion from each sub experiment and then get our best model with all of thesel, but the thing is we cant really get the best model before doing inverse dynamics that is before doing the medical velidation using myosuite, so lets keep the best model till the end.
  Chapter 4   Generative Model: CVAE     
    4.1  Evaluation Framework               
    4.2  Shared Specifications                              
    4.3  Loss Function                        
    4.4  Architectural Development (Phase 1)                    
    4.5  Component Ablation Study (Phase 2)   
    4.6  Physical Plausibility Constraints (Phase 3)              
      4.6.1  Final Model Loss Function        
      4.6.2  Phase 3 Results                                      
    4.7  Hyperparameter Optimisation (Phase 4) 
      4.7.1  Guidance Scale (E-phase)                                 
      4.7.2  Conditioning Dropout (F-phase)                      
      4.7.3  Epoch Count (G-phase)            
    4.8  Augmentation Selection & Inference Averaging (Phase 5)
      4.8.1  Augmentation Method Comparison                         
      4.8.2  Multi-Sample Averaging           
    4.9  Final Model Selection                                
    4.10 Motion Generation Pipeline          

now we take our model sthat generates the motion and experiment on them all and all of our results here and do if as well and finally finalise our best model, actually im quite confused on how to do this, cause we also have a results section, so maybe the 3 chaps, dataset and augmentation, generative model and sim pipeline just talk about the stuff we did and not the actual results which i dont think would be right, we would have a lot of results in these 3 sections but they wont be the final results but will contribute to the final model so its imp to have there to like show our scientific reasoning properly, and after we get our best model from these experiments we can move on to results where we actually showcase the final results
  Chapter 5   Biomechanical Simulation Pipeline
    5.1  Inverse Kinematics                
      5.1.1  Pipeline                                            
      5.1.2  V_Vector Reconstruction      
      5.1.3  Coordinate Frame Alignment                       
      5.1.4  IK Solver                    
    5.2  Inverse Dynamics                                    
      5.2.1  Overview                         
      5.2.2  Equality Constraint Forces                        
      5.2.3  Scapulohumeral Rhythm      
      5.2.4  Anatomical Masking                                    
      5.2.5  Static Optimisation             
      5.2.6  Spasticity Co-Contraction Model                        
    5.3  Effort Metrics                                        
      5.3.1  Activation-Time Integral         
      5.3.2  Co-Contraction Index                                  
      5.3.3  Torque-ROM Ratio               
    5.4  Muscle Synergy Analysis                                   
these results is like a final curtain over everyting, could be like a graph with everything in them or could just target the final model alone and havin either or both is nice.
  Chapter 6   Results                                   
    6.1  Preliminary Kinematic Analysis      
    6.2  Experimental Configuration       
    6.3  Descriptive Statistics                         
    6.4  Correlation Analysis               
    6.5  Group Comparisons and Statistical Tests             
    6.6  Joint Torque Validation          
    6.7  CCI by Severity Group                                     
    6.8  Muscle Synergy Results            
    6.9  Phase-Specific Muscle Dominance                            
    6.10 Literature Validation Summary                          
   simply conclude the experiment,
  Chapter 7   Conclusions                                     
    7.1  Summary of Findings             
    7.2  Problems Encountered             
    7.3  Limitations and Future Work                               
    7.4  Recommendations for Future Work      
    
  Bibliography                                
  Appendix A  End-to-End Pipeline Architecture
  Appendix B  Supervisor Meeting Minutes        