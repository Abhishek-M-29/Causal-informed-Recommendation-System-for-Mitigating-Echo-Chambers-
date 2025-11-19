import pandas as pd
import numpy as np
import os

def create_mock_mind_data(data_path):
    """
    Creates dummy news.tsv and behaviors.tsv for testing.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    news_path = os.path.join(data_path, 'news.tsv')
    behaviors_path = os.path.join(data_path, 'behaviors.tsv')
    
    if os.path.exists(news_path) and os.path.exists(behaviors_path):
        print("Mock data already exists.")
        return

    print("Creating mock MIND data...")
    
    # Create News Data
    # NewsID, Category, SubCategory, Title, Abstract, URL, TitleEntities, AbstractEntities
    news_ids = [f'N{i}' for i in range(100)]
    categories = ['news', 'sports', 'finance', 'entertainment', 'tech']
    
    news_data = []
    for nid in news_ids:
        cat = np.random.choice(categories)
        news_data.append([
            nid, cat, f'sub_{cat}', f'Title for {nid}', f'Abstract for {nid}', 
            f'http://url/{nid}', '[]', '[]'
        ])
        
    pd.DataFrame(news_data).to_csv(news_path, sep='\t', header=False, index=False)
    
    # Create Behaviors Data
    # ImpressionID, UserID, Time, History, Impressions
    behaviors_data = []
    for i in range(200):
        uid = f'U{np.random.randint(0, 50)}'
        history = ' '.join(np.random.choice(news_ids, size=np.random.randint(0, 10), replace=False))
        
        # Impressions: N1-0 N2-1 ...
        imps = []
        candidates = np.random.choice(news_ids, size=10, replace=False)
        for cand in candidates:
            label = np.random.choice([0, 1], p=[0.9, 0.1])
            imps.append(f'{cand}-{label}')
            
        behaviors_data.append([
            i, uid, '11/11/2019 9:00:00 AM', history, ' '.join(imps)
        ])
        
    pd.DataFrame(behaviors_data).to_csv(behaviors_path, sep='\t', header=False, index=False)
    print("Mock data created.")
