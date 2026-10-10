from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

class UserAuthTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = reverse('accounts:register')
        self.login_url = reverse('accounts:login')
        self.logout_url = reverse('accounts:logout')
        self.profile_url = reverse('accounts:profile')
        
        self.user_data = {
            'username': 'teststudent',
            'first_name': 'Test',
            'last_name': 'Student',
            'email': 'student@example.com',
            'password1': 'StrongPass123!',
            'password2': 'StrongPass123!',
        }

    def test_user_registration(self):
        """Test user registration with valid data and secure password hashing"""
        response = self.client.post(self.register_url, self.user_data, follow=True)
        self.assertEqual(response.status_code, 200)
        
        # User created in database
        user = User.objects.filter(username='teststudent').first()
        self.assertIsNotNone(user)
        self.assertEqual(user.email, 'student@example.com')
        
        # Password must be hashed, not stored as plain text
        self.assertNotEqual(user.password, 'StrongPass123!')
        self.assertTrue(user.check_password('StrongPass123!'))

    def test_user_login_and_logout(self):
        """Test logging in and logging out"""
        user = User.objects.create_user(
            username='johndoe',
            email='john@example.com',
            password='Password123!'
        )
        
        # Login with correct credentials
        response = self.client.post(self.login_url, {
            'username': 'johndoe',
            'password': 'Password123!'
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['user'].is_authenticated)

        # Logout
        logout_response = self.client.get(self.logout_url, follow=True)
        self.assertEqual(logout_response.status_code, 200)
        self.assertFalse(logout_response.context['user'].is_authenticated)

    def test_protected_profile_page_redirects_unauthenticated_user(self):
        """Test accessing profile page when logged out redirects to login page"""
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith(self.login_url))
        self.assertIn('next=/accounts/profile/', response.url)

    def test_authenticated_user_can_access_and_update_profile(self):
        """Test authenticated user can access profile and update their details"""
        user = User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='Password123!'
        )
        self.client.login(username='alice', password='Password123!')

        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'alice')

        # Update profile
        update_response = self.client.post(self.profile_url, {
            'first_name': 'AliceUpdated',
            'last_name': 'Smith',
            'email': 'alicenew@example.com',
        }, follow=True)
        self.assertEqual(update_response.status_code, 200)
        
        user.refresh_from_db()
        self.assertEqual(user.first_name, 'AliceUpdated')
        self.assertEqual(user.email, 'alicenew@example.com')

    def test_registration_validation_errors(self):
        """Test registration errors for mismatched passwords and short passwords."""
        # Mismatched passwords
        bad_data = self.user_data.copy()
        bad_data['password2'] = 'DifferentPassword123!'
        response = self.client.post(self.register_url, bad_data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='teststudent').exists())

        # Too short password
        short_data = self.user_data.copy()
        short_data['password1'] = '123'
        short_data['password2'] = '123'
        response = self.client.post(self.register_url, short_data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='teststudent').exists())

    def test_duplicate_username_registration_rejected(self):
        """Test registration rejects duplicate usernames."""
        User.objects.create_user(username='existinguser', password='Password123!')
        dup_data = self.user_data.copy()
        dup_data['username'] = 'existinguser'
        response = self.client.post(self.register_url, dup_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(User.objects.filter(username='existinguser').count(), 1)
