 Running the pipeline                                               

  # Install dependencies
  pip install -r requirements.txt
  playwright install chromium

  # Full run (all ~5,572 PNG screenshots)                                                       
  python pipeline.py
                                                                                                
  # Test with only 20 screenshots                                                               
  python pipeline.py --limit 20
                                                                                                
  # Metadata + HTML only, skip Playwright                            
  python pipeline.py --skip-screenshots
                                                                                                
  # Verify balance after any run
  python verify_balance.py                                                                      
                                                                     
  ---
  Design notes
              
  - Deduplication: 18,000 respondent rows map to 5,572 unique stimuli (avg 3.2 respondents share
   each image), keeping screenshot rendering manageable.                                        
  - Balance strategy: for each topic, the 36 non-topic factor combinations are tiled and
  shuffled independently, so respondent i draws row i from every topic's pool — guaranteeing    
  perfect within-respondent topic coverage and near-perfect marginal balance across the full
  sample.                                                                                       
  - Reproducibility: all randomness flows through a single np.random.default_rng(seed) instance;
   pass --seed N to change.                                                                     
   
✻ Brewed for 18m 55s · 1 local agent still running                                              
                                                                     
⏺ Agent "Save project memory" completed                                                         
                                                                                       
⏺ Let me know if you'd like to adjust anything — for example, adding more profiles per cell,    
  tweaking the Instagram visual design, modifying the comment pool, or integrating the pipeline
  with a survey platform like Qualtrics. 