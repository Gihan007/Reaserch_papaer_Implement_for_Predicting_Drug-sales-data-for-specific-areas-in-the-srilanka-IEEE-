import re

# New navigation HTML
NEW_NAV = '''    <!-- Navigation -->
    <nav class="navbar">
        <div class="nav-container">
            <div class="nav-brand">
                <i class="fas fa-heartbeat"></i>
                <span>PharmaPredictAI</span>
            </div>
            <div class="nav-menu">
                <a href="/" class="nav-link{dash_active}">
                    <i class="fas fa-home"></i>
                    <span>Dashboard</span>
                </a>
                <a href="/forecast" class="nav-link{forecast_active}">
                    <i class="fas fa-chart-line"></i>
                    <span>Forecasting</span>
                </a>
                <div class="nav-dropdown">
                    <a href="#" class="nav-link{ai_active}">
                        <i class="fas fa-brain"></i>
                        <span>AI Models</span>
                        <i class="fas fa-chevron-down"></i>
                    </a>
                    <div class="dropdown-menu">
                        <a href="/meta-learning" class="dropdown-item">
                            <i class="fas fa-brain"></i>
                            <span>Meta Learning</span>
                        </a>
                        <a href="/advanced" class="dropdown-item">
                            <i class="fas fa-robot"></i>
                            <span>Advanced AI</span>
                        </a>
                        <a href="/causal" class="dropdown-item">
                            <i class="fas fa-project-diagram"></i>
                            <span>Causal Analysis</span>
                        </a>
                    </div>
                </div>
                <a href="/analytics" class="nav-link{analytics_active}">
                    <i class="fas fa-chart-bar"></i>
                    <span>Analytics</span>
                </a>
            </div>
            <div class="nav-actions">
                <button class="btn-icon" id="notificationBtn" title="Notifications">
                    <i class="fas fa-bell"></i>
                    <span class="badge">3</span>
                </button>
                <div class="user-dropdown">
                    <button class="user-profile">
                        <img src="https://ui-avatars.com/api/?name=Admin&background=0D8ABC&color=fff" alt="User">
                        <span class="user-name">Admin</span>
                        <i class="fas fa-chevron-down"></i>
                    </button>
                    <div class="dropdown-menu dropdown-menu-right">
                        <a href="#" class="dropdown-item">
                            <i class="fas fa-user"></i>
                            <span>Profile</span>
                        </a>
                        <a href="#" class="dropdown-item">
                            <i class="fas fa-cog"></i>
                            <span>Settings</span>
                        </a>
                        <div class="dropdown-divider"></div>
                        <a href="#" class="dropdown-item">
                            <i class="fas fa-sign-out-alt"></i>
                            <span>Logout</span>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </nav>'''

files_to_update = {
    'templates/meta-learning.html': {'dash_active': '', 'forecast_active': '', 'ai_active': ' active', 'analytics_active': ''},
    'templates/advanced.html': {'dash_active': '', 'forecast_active': '', 'ai_active': ' active', 'analytics_active': ''},
    'templates/causal.html': {'dash_active': '', 'forecast_active': '', 'ai_active': ' active', 'analytics_active': ''},
    'templates/analytics.html': {'dash_active': '', 'forecast_active': '', 'ai_active': '', 'analytics_active': ' active'},
}

for file_path, active_states in files_to_update.items():
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace navigation section
    pattern = r'    <!-- Navigation -->.*?</nav>'
    nav_html = NEW_NAV.format(**active_states)
    new_content = re.sub(pattern, nav_html, content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Updated: {file_path}")

print("All navigation bars updated successfully!")
