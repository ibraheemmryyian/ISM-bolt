def match_buyers_sellers(buyers, sellers):
    """
    Simple matching algorithm for MVP
    Args:
        buyers: List of buyer profiles with needs
        sellers: List of seller profiles with capabilities
    Returns:
        List of matches (buyer_id, seller_id, match_score)
    """
    matches = []
    for buyer in buyers:
        for seller in sellers:
            # Simple scoring based on keyword overlap
            buyer_needs = set(buyer['needs'].lower().split())
            seller_caps = set(seller['capabilities'].lower().split())
            common = buyer_needs.intersection(seller_caps)
            score = len(common) / max(len(buyer_needs), 1)
            
            if score > 0.2:  # Minimum threshold
                matches.append({
                    'buyer_id': buyer['id'],
                    'seller_id': seller['id'],
                    'score': round(score, 2),
                    'matched_keywords': list(common)
                })
    
    # Sort by best matches first
    return sorted(matches, key=lambda x: x['score'], reverse=True)

# Example usage for testing
if __name__ == "__main__":
    buyers = [
        {'id': 'b1', 'needs': 'machine learning computer vision'},
        {'id': 'b2', 'needs': 'predictive maintenance IoT'}
    ]
    
    sellers = [
        {'id': 's1', 'capabilities': 'AI computer vision deep learning'},
        {'id': 's2', 'capabilities': 'IoT sensors predictive analytics'}
    ]
    
    matches = match_buyers_sellers(buyers, sellers)
    print("Matches:", matches)
