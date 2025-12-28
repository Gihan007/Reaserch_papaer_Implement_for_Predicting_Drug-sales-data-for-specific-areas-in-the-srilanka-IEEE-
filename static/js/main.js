// ============================================
// PharmaPredictAI - Main JavaScript
// ============================================

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeNotifications();
    animateStats();
});

// ============================================
// Navigation
// ============================================
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const currentPath = window.location.pathname;
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// ============================================
// Notifications
// ============================================
function initializeNotifications() {
    const notificationBtn = document.getElementById('notificationBtn');
    if (notificationBtn) {
        notificationBtn.addEventListener('click', () => {
            showNotification('You have 3 new notifications', 'info');
        });
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)}"></i>
        <span>${message}</span>
        <button class="notification-close">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    });
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Add notification styles dynamically
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    .notification {
        position: fixed;
        top: 90px;
        right: 20px;
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        display: flex;
        align-items: center;
        gap: 1rem;
        z-index: 9999;
        transform: translateX(400px);
        transition: transform 0.3s ease;
        min-width: 300px;
        max-width: 400px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-success {
        border-left: 4px solid #28A745;
    }
    
    .notification-error {
        border-left: 4px solid #DC3545;
    }
    
    .notification-warning {
        border-left: 4px solid #FFC107;
    }
    
    .notification-info {
        border-left: 4px solid #17A2B8;
    }
    
    .notification i {
        font-size: 1.25rem;
    }
    
    .notification-success i { color: #28A745; }
    .notification-error i { color: #DC3545; }
    .notification-warning i { color: #FFC107; }
    .notification-info i { color: #17A2B8; }
    
    .notification span {
        flex: 1;
    }
    
    .notification-close {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: #6C757D;
        padding: 0;
        width: 24px;
        height: 24px;
    }
    
    .notification-close:hover {
        color: #2C3E50;
    }
`;
document.head.appendChild(notificationStyles);

// ============================================
// Stats Animation
// ============================================
function animateStats() {
    const statNumbers = document.querySelectorAll('.stat-number');
    
    statNumbers.forEach(stat => {
        const target = parseInt(stat.textContent.replace(/,/g, ''));
        if (!isNaN(target)) {
            animateValue(stat, 0, target, 2000);
        }
    });
}

function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current).toLocaleString();
    }, 16);
}

// ============================================
// Tabs Management
// ============================================
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Show corresponding tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}Tab`).classList.add('active');
    });
});

// ============================================
// API Helper Functions
// ============================================
async function apiCall(url, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    if (data && method !== 'GET') {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        
        // Try to parse JSON response
        let result;
        try {
            result = await response.json();
        } catch (parseError) {
            console.error('Failed to parse JSON response:', parseError);
            return {
                success: false,
                status: 'error',
                message: 'Invalid response from server'
            };
        }
        
        // Log response for debugging
        console.log(`API ${method} ${url}:`, response.status, result);
        
        // If HTTP status is not OK, ensure result reflects error
        if (!response.ok && !result.status) {
            result.status = 'error';
            result.success = false;
        }
        
        return result;
    } catch (error) {
        console.error('API Error:', error);
        showNotification('Network error. Please check your connection.', 'error');
        return {
            success: false,
            status: 'error',
            message: error.message || 'Network error occurred'
        };
    }
}

// ============================================
// Utility Functions
// ============================================
function formatNumber(num, decimals = 2) {
    return num.toFixed(decimals);
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function showLoading(element) {
    if (element) {
        element.style.display = 'flex';
    }
}

function hideLoading(element) {
    if (element) {
        element.style.display = 'none';
    }
}

// ============================================
// Drug Categories Carousel Animation
// ============================================
function initDrugCategoriesCarousel() {
    const indicators = document.querySelectorAll('.carousel-indicators .indicator');
    const categoryCards = document.querySelectorAll('.category-card');
    
    if (!indicators.length || !categoryCards.length) return;
    
    let currentSlide = 0;
    let autoPlayInterval;
    
    function showSlide(index) {
        indicators.forEach((ind, i) => {
            ind.classList.toggle('active', i === index);
        });
        
        // Highlight the corresponding category
        categoryCards.forEach((card, i) => {
            if (i === index) {
                card.style.transform = 'translateY(-12px) scale(1.05)';
                card.style.boxShadow = '0 20px 50px rgba(13, 138, 188, 0.3)';
            } else {
                card.style.transform = 'translateY(0) scale(1)';
                card.style.boxShadow = '0 8px 24px rgba(0,0,0,0.12)';
            }
        });
    }
    
    function nextSlide() {
        currentSlide = (currentSlide + 1) % indicators.length;
        showSlide(currentSlide);
    }
    
    function startAutoPlay() {
        autoPlayInterval = setInterval(nextSlide, 3500);
    }
    
    function stopAutoPlay() {
        clearInterval(autoPlayInterval);
    }
    
    // Indicator click handlers
    indicators.forEach((indicator, index) => {
        indicator.addEventListener('click', () => {
            currentSlide = index;
            showSlide(currentSlide);
            stopAutoPlay();
            startAutoPlay(); // Restart auto-play
        });
    });
    
    // Card hover effects
    categoryCards.forEach((card, index) => {
        card.addEventListener('mouseenter', () => {
            stopAutoPlay();
        });
        
        card.addEventListener('mouseleave', () => {
            startAutoPlay();
        });
    });
    
    // Start auto-play
    startAutoPlay();
    
    // Intersection Observer for scroll-triggered animations
    const observerOptions = {
        threshold: 0.2,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
            }
        });
    }, observerOptions);
    
    categoryCards.forEach(card => observer.observe(card));
}

// Initialize on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDrugCategoriesCarousel);
} else {
    initDrugCategoriesCarousel();
}

// Export functions for use in other files
window.PharmaPredictAI = {
    apiCall,
    showNotification,
    formatNumber,
    formatDate,
    showLoading,
    hideLoading
};
