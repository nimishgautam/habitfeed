"""
Dummy article content for testing embedding and recommendation logic.
Each article has different content to ensure they produce different embeddings.
"""

DUMMY_ARTICLES = {
    # Technology articles - should cluster together in embedding space
    "tech_ai": {
        "title": "Advances in Artificial Intelligence and Machine Learning",
        "content": """
        Artificial intelligence and machine learning have made remarkable strides in recent years. 
        From natural language processing to computer vision, AI systems are becoming increasingly 
        sophisticated. Deep learning neural networks are now capable of understanding complex 
        patterns in data, enabling breakthrough applications in healthcare, autonomous vehicles, 
        and scientific research. The transformer architecture has revolutionized language models, 
        making them more coherent and contextually aware. As we continue to push the boundaries 
        of what's possible with AI, we must also consider the ethical implications and ensure 
        responsible development of these powerful technologies.
        """,
        "url": "https://tech.example.com/ai-advances",
        "description": "Latest developments in AI and ML technologies"
    },
    
    "tech_quantum": {
        "title": "Quantum Computing Breakthrough: New Algorithms and Applications",
        "content": """
        Quantum computing represents a paradigm shift in computational power and capability. 
        Recent breakthroughs in quantum algorithms have opened new possibilities for solving 
        complex optimization problems, cryptography, and scientific simulations. Quantum 
        supremacy demonstrations have shown that quantum computers can outperform classical 
        computers for specific tasks. Major tech companies are investing heavily in quantum 
        research, developing more stable qubits and error correction methods. The potential 
        applications span from drug discovery to financial modeling, promising to revolutionize 
        how we approach computationally intensive problems.
        """,
        "url": "https://tech.example.com/quantum-computing",
        "description": "Quantum computing breakthroughs and applications"
    },
    
    # Health and medical articles
    "health_nutrition": {
        "title": "The Science of Nutrition: Understanding Micronutrients and Health",
        "content": """
        Proper nutrition plays a crucial role in maintaining optimal health and preventing 
        chronic diseases. Micronutrients such as vitamins and minerals are essential for 
        cellular function, immune system support, and metabolic processes. Recent research 
        has highlighted the importance of vitamin D for bone health and immune function, 
        while omega-3 fatty acids have been linked to cardiovascular and brain health. 
        A balanced diet rich in fruits, vegetables, whole grains, and lean proteins provides 
        the necessary nutrients for optimal wellbeing. Understanding the bioavailability 
        of nutrients and their interactions helps optimize dietary choices for better health outcomes.
        """,
        "url": "https://health.example.com/nutrition-science",
        "description": "Scientific insights into nutrition and health"
    },
    
    "health_exercise": {
        "title": "Exercise Physiology: How Physical Activity Transforms the Body",
        "content": """
        Regular physical exercise induces profound physiological adaptations that enhance 
        health and performance. Cardiovascular exercise strengthens the heart muscle, improves 
        circulation, and increases oxygen delivery to tissues. Resistance training builds 
        muscle mass, increases bone density, and boosts metabolic rate. Exercise also has 
        powerful effects on mental health, releasing endorphins and reducing stress hormones. 
        The molecular mechanisms underlying exercise benefits include improved insulin sensitivity, 
        enhanced mitochondrial function, and increased production of growth factors. 
        Understanding exercise physiology helps optimize training programs for specific health 
        and fitness goals.
        """,
        "url": "https://health.example.com/exercise-physiology",
        "description": "How exercise transforms the human body"
    },
    
    # Environmental and climate articles
    "environment_climate": {
        "title": "Climate Change Impacts: Global Warming and Ecosystem Disruption",
        "content": """
        Climate change is causing unprecedented disruptions to global ecosystems and weather 
        patterns. Rising temperatures are melting polar ice caps, causing sea levels to rise 
        and threatening coastal communities. Extreme weather events such as hurricanes, droughts, 
        and floods are becoming more frequent and severe. These changes are disrupting wildlife 
        habitats, migration patterns, and biodiversity. The agricultural sector faces challenges 
        from changing precipitation patterns and increased pest pressure. Urgent action is needed 
        to reduce greenhouse gas emissions and implement adaptation strategies to mitigate the 
        worst effects of climate change on both human societies and natural ecosystems.
        """,
        "url": "https://environment.example.com/climate-impacts",
        "description": "Global climate change impacts and consequences"
    },
    
    "environment_renewable": {
        "title": "Renewable Energy Revolution: Solar, Wind, and Storage Technologies",
        "content": """
        The renewable energy sector is experiencing rapid growth and technological advancement. 
        Solar photovoltaic technology has become increasingly efficient and cost-effective, 
        making it competitive with fossil fuels in many markets. Wind energy systems are 
        becoming larger and more powerful, with offshore installations offering tremendous 
        potential. Energy storage technologies, particularly lithium-ion batteries, are 
        solving the intermittency challenges of renewable sources. Smart grid technologies 
        are enabling better integration of distributed renewable energy sources. Government 
        policies and corporate commitments are accelerating the transition to clean energy, 
        creating a sustainable future for energy production and consumption.
        """,
        "url": "https://environment.example.com/renewable-energy",
        "description": "Advances in renewable energy technologies"
    },
    
    # Finance and economics articles
    "finance_markets": {
        "title": "Understanding Financial Markets: Stocks, Bonds, and Investment Strategies",
        "content": """
        Financial markets play a crucial role in allocating capital and enabling economic 
        growth. Stock markets provide companies with access to capital while offering 
        investors opportunities for wealth building. Bond markets facilitate government 
        and corporate borrowing for infrastructure and business expansion. Understanding 
        market dynamics, risk assessment, and diversification principles is essential 
        for successful investing. Modern portfolio theory emphasizes the importance of 
        asset allocation and risk management. Economic indicators such as inflation, 
        interest rates, and GDP growth significantly influence market performance and 
        investment decisions.
        """,
        "url": "https://finance.example.com/market-analysis",
        "description": "Financial market analysis and investment strategies"
    },
    
    "finance_crypto": {
        "title": "Cryptocurrency and Blockchain: Digital Finance Revolution",
        "content": """
        Cryptocurrency and blockchain technology are transforming the financial landscape. 
        Bitcoin pioneered decentralized digital currency, demonstrating the potential 
        for peer-to-peer transactions without intermediaries. Ethereum introduced smart 
        contracts, enabling programmable money and decentralized applications. Blockchain 
        technology offers transparency, immutability, and security for various applications 
        beyond currency. Central bank digital currencies (CBDCs) are being explored by 
        governments worldwide. The crypto ecosystem includes decentralized finance (DeFi) 
        protocols that enable lending, borrowing, and trading without traditional banks. 
        Regulatory frameworks are evolving to address the challenges and opportunities 
        presented by digital assets.
        """,
        "url": "https://finance.example.com/cryptocurrency",
        "description": "Cryptocurrency and blockchain technology trends"
    }
} 