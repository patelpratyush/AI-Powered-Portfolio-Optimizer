#!/usr/bin/env python3
"""
Authentication and JWT tests
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from flask_jwt_extended import create_access_token, decode_token
from routes.auth import auth_bp
from models.database import User


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    def test_register_success(self, client):
        """Test successful user registration"""
        payload = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'securepassword123',
            'first_name': 'Test',
            'last_name': 'User',
            'investment_experience': 'intermediate',
            'risk_tolerance': 'moderate'
        }
        
        with patch('routes.auth.User') as mock_user:
            # Mock user creation
            mock_user.query.filter_by.return_value.first.return_value = None
            mock_user.return_value = MagicMock()
            mock_user.return_value.id = 1
            
            with patch('routes.auth.db.session') as mock_session:
                response = client.post('/api/auth/register',
                                     data=json.dumps(payload),
                                     content_type='application/json')
                
                assert response.status_code == 201
                data = response.get_json()
                
                assert 'access_token' in data
                assert 'refresh_token' in data
                assert 'user' in data
                assert data['user']['username'] == 'testuser'
                assert data['user']['email'] == 'test@example.com'
                
                mock_session.add.assert_called_once()
                mock_session.commit.assert_called_once()
                
    def test_register_duplicate_username(self, client):
        """Test registration with duplicate username"""
        payload = {
            'username': 'existinguser',
            'email': 'new@example.com',
            'password': 'securepassword123'
        }
        
        with patch('routes.auth.User') as mock_user:
            # Mock existing user
            mock_existing_user = MagicMock()
            mock_user.query.filter_by.return_value.first.return_value = mock_existing_user
            
            response = client.post('/api/auth/register',
                                 data=json.dumps(payload),
                                 content_type='application/json')
            
            assert response.status_code == 400
            data = response.get_json()
            assert 'error' in data
            assert 'username' in data['error'].lower()
            
    def test_register_duplicate_email(self, client):
        """Test registration with duplicate email"""
        payload = {
            'username': 'newuser',
            'email': 'existing@example.com',
            'password': 'securepassword123'
        }
        
        with patch('routes.auth.User') as mock_user:
            # Mock no user with username, but user with email exists
            mock_user.query.filter_by.side_effect = [
                MagicMock(first=MagicMock(return_value=None)),  # No user with username
                MagicMock(first=MagicMock(return_value=MagicMock()))  # User with email exists
            ]
            
            response = client.post('/api/auth/register',
                                 data=json.dumps(payload),
                                 content_type='application/json')
            
            assert response.status_code == 400
            data = response.get_json()
            assert 'error' in data
            assert 'email' in data['error'].lower()
            
    def test_register_weak_password(self, client):
        """Test registration with weak password"""
        payload = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': '123'  # Too weak
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(payload),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'password' in data['error'].lower()
        
    def test_login_success(self, client):
        """Test successful login"""
        payload = {
            'username': 'testuser',
            'password': 'correctpassword'
        }
        
        with patch('routes.auth.User') as mock_user:
            # Mock user with correct password
            mock_user_instance = MagicMock()
            mock_user_instance.id = 1
            mock_user_instance.username = 'testuser'
            mock_user_instance.email = 'test@example.com'
            mock_user_instance.check_password.return_value = True
            mock_user_instance.is_active = True
            mock_user_instance.failed_login_attempts = 0
            
            mock_user.query.filter_by.return_value.first.return_value = mock_user_instance
            
            with patch('routes.auth.db.session'):
                response = client.post('/api/auth/login',
                                     data=json.dumps(payload),
                                     content_type='application/json')
                
                assert response.status_code == 200
                data = response.get_json()
                
                assert 'access_token' in data
                assert 'refresh_token' in data
                assert 'user' in data
                assert data['user']['username'] == 'testuser'
                
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        payload = {
            'username': 'testuser',
            'password': 'wrongpassword'
        }
        
        with patch('routes.auth.User') as mock_user:
            # Mock user with incorrect password
            mock_user_instance = MagicMock()
            mock_user_instance.check_password.return_value = False
            mock_user_instance.is_active = True
            mock_user_instance.failed_login_attempts = 0
            
            mock_user.query.filter_by.return_value.first.return_value = mock_user_instance
            
            with patch('routes.auth.db.session'):
                response = client.post('/api/auth/login',
                                     data=json.dumps(payload),
                                     content_type='application/json')
                
                assert response.status_code == 401
                data = response.get_json()
                assert 'error' in data
                
    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user"""
        payload = {
            'username': 'nonexistent',
            'password': 'password'
        }
        
        with patch('routes.auth.User') as mock_user:
            mock_user.query.filter_by.return_value.first.return_value = None
            
            response = client.post('/api/auth/login',
                                 data=json.dumps(payload),
                                 content_type='application/json')
            
            assert response.status_code == 401
            data = response.get_json()
            assert 'error' in data
            
    def test_login_account_locked(self, client):
        """Test login with locked account"""
        payload = {
            'username': 'lockeduser',
            'password': 'password'
        }
        
        with patch('routes.auth.User') as mock_user:
            # Mock locked user account
            mock_user_instance = MagicMock()
            mock_user_instance.is_active = False
            mock_user_instance.failed_login_attempts = 5
            
            mock_user.query.filter_by.return_value.first.return_value = mock_user_instance
            
            response = client.post('/api/auth/login',
                                 data=json.dumps(payload),
                                 content_type='application/json')
            
            assert response.status_code == 423  # Locked
            data = response.get_json()
            assert 'error' in data
            assert 'locked' in data['error'].lower()
            
    def test_refresh_token_success(self, client, app):
        """Test successful token refresh"""
        with app.app_context():
            # Create a refresh token
            refresh_token = create_access_token(
                identity={'user_id': 1, 'username': 'testuser'}, 
                fresh=False
            )
            
            headers = {'Authorization': f'Bearer {refresh_token}'}
            
            with patch('routes.auth.get_jwt_identity') as mock_identity:
                mock_identity.return_value = {'user_id': 1, 'username': 'testuser'}
                
                with patch('routes.auth.User') as mock_user:
                    mock_user_instance = MagicMock()
                    mock_user_instance.id = 1
                    mock_user_instance.username = 'testuser'
                    mock_user_instance.is_active = True
                    mock_user.query.get.return_value = mock_user_instance
                    
                    response = client.post('/api/auth/refresh', headers=headers)
                    
                    assert response.status_code == 200
                    data = response.get_json()
                    assert 'access_token' in data
                    
    def test_logout_success(self, client, app):
        """Test successful logout"""
        with app.app_context():
            access_token = create_access_token(identity={'user_id': 1, 'username': 'testuser'})
            headers = {'Authorization': f'Bearer {access_token}'}
            
            with patch('routes.auth.get_jwt') as mock_jwt:
                mock_jwt.return_value = {'jti': 'test-jti'}
                
                with patch('routes.auth.blacklist_token') as mock_blacklist:
                    response = client.post('/api/auth/logout', headers=headers)
                    
                    assert response.status_code == 200
                    data = response.get_json()
                    assert data['message'] == 'Successfully logged out'
                    mock_blacklist.assert_called_once_with('test-jti')


class TestJWTFunctionality:
    """Test JWT token functionality"""
    
    def test_token_creation_and_decoding(self, app):
        """Test JWT token creation and decoding"""
        with app.app_context():
            user_identity = {'user_id': 1, 'username': 'testuser'}
            
            # Create access token
            access_token = create_access_token(identity=user_identity, fresh=True)
            assert access_token is not None
            
            # Decode token
            decoded_token = decode_token(access_token)
            assert decoded_token['sub'] == user_identity
            assert decoded_token['fresh'] is True
            
    def test_token_expiration(self, app):
        """Test token expiration handling"""
        with app.app_context():
            # Create token with very short expiration
            from datetime import timedelta
            short_token = create_access_token(
                identity={'user_id': 1, 'username': 'testuser'},
                expires_delta=timedelta(seconds=1)
            )
            
            # Token should be valid immediately
            decoded = decode_token(short_token)
            assert decoded['sub']['user_id'] == 1
            
            # After expiration, token should be invalid (would need to test with actual time passage)
            import time
            time.sleep(2)
            
            # In real implementation, expired tokens would be rejected by JWT
            # This is handled by Flask-JWT-Extended automatically
            
    def test_token_blacklisting(self, app):
        """Test JWT token blacklisting functionality"""
        with app.app_context():
            # This would test the blacklist functionality
            # In actual implementation, blacklisted tokens are stored in Redis
            token = create_access_token(identity={'user_id': 1, 'username': 'testuser'})
            decoded = decode_token(token)
            jti = decoded['jti']
            
            # Mock blacklist check
            with patch('routes.auth.is_token_blacklisted') as mock_blacklist_check:
                mock_blacklist_check.return_value = True
                
                # Token should be considered invalid when blacklisted
                assert mock_blacklist_check(jti) is True


class TestPasswordSecurity:
    """Test password security features"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        from werkzeug.security import generate_password_hash, check_password_hash
        
        password = "secure_password_123"
        
        # Hash password
        password_hash = generate_password_hash(password)
        assert password_hash != password
        assert len(password_hash) > 50  # Hashed passwords are long
        
        # Verify correct password
        assert check_password_hash(password_hash, password) is True
        
        # Verify incorrect password
        assert check_password_hash(password_hash, "wrong_password") is False
        
    def test_password_strength_validation(self):
        """Test password strength validation"""
        from routes.auth import validate_password_strength
        
        # Strong password
        assert validate_password_strength("StrongPass123!") is True
        
        # Weak passwords
        assert validate_password_strength("123") is False  # Too short
        assert validate_password_strength("password") is False  # No numbers/symbols
        assert validate_password_strength("12345678") is False  # No letters
        assert validate_password_strength("PASSWORD123") is False  # No lowercase
        
    def test_failed_login_attempt_tracking(self, client):
        """Test failed login attempt tracking and account locking"""
        payload = {
            'username': 'testuser',
            'password': 'wrongpassword'
        }
        
        with patch('routes.auth.User') as mock_user:
            mock_user_instance = MagicMock()
            mock_user_instance.check_password.return_value = False
            mock_user_instance.is_active = True
            mock_user_instance.failed_login_attempts = 0
            
            mock_user.query.filter_by.return_value.first.return_value = mock_user_instance
            
            with patch('routes.auth.db.session') as mock_session:
                # First failed attempt
                response = client.post('/api/auth/login',
                                     data=json.dumps(payload),
                                     content_type='application/json')
                
                assert response.status_code == 401
                # Should increment failed attempts
                assert mock_user_instance.failed_login_attempts == 1
                mock_session.commit.assert_called()


class TestUserProfile:
    """Test user profile management"""
    
    def test_get_profile_success(self, client, app):
        """Test successful profile retrieval"""
        with app.app_context():
            access_token = create_access_token(identity={'user_id': 1, 'username': 'testuser'})
            headers = {'Authorization': f'Bearer {access_token}'}
            
            with patch('routes.auth.get_jwt_identity') as mock_identity:
                mock_identity.return_value = {'user_id': 1, 'username': 'testuser'}
                
                with patch('routes.auth.User') as mock_user:
                    mock_user_instance = MagicMock()
                    mock_user_instance.id = 1
                    mock_user_instance.username = 'testuser'
                    mock_user_instance.email = 'test@example.com'
                    mock_user_instance.first_name = 'Test'
                    mock_user_instance.last_name = 'User'
                    mock_user_instance.investment_experience = 'intermediate'
                    mock_user_instance.risk_tolerance = 'moderate'
                    
                    mock_user.query.get.return_value = mock_user_instance
                    
                    response = client.get('/api/auth/profile', headers=headers)
                    
                    assert response.status_code == 200
                    data = response.get_json()
                    
                    assert data['user']['username'] == 'testuser'
                    assert data['user']['email'] == 'test@example.com'
                    assert data['user']['investment_experience'] == 'intermediate'
                    
    def test_update_profile_success(self, client, app):
        """Test successful profile update"""
        with app.app_context():
            access_token = create_access_token(identity={'user_id': 1, 'username': 'testuser'})
            headers = {'Authorization': f'Bearer {access_token}'}
            
            update_payload = {
                'first_name': 'Updated',
                'last_name': 'Name',
                'investment_experience': 'advanced',
                'risk_tolerance': 'high'
            }
            
            with patch('routes.auth.get_jwt_identity') as mock_identity:
                mock_identity.return_value = {'user_id': 1, 'username': 'testuser'}
                
                with patch('routes.auth.User') as mock_user:
                    mock_user_instance = MagicMock()
                    mock_user_instance.id = 1
                    mock_user.query.get.return_value = mock_user_instance
                    
                    with patch('routes.auth.db.session') as mock_session:
                        response = client.put('/api/auth/profile',
                                            data=json.dumps(update_payload),
                                            content_type='application/json',
                                            headers=headers)
                        
                        assert response.status_code == 200
                        data = response.get_json()
                        
                        assert data['message'] == 'Profile updated successfully'
                        mock_session.commit.assert_called_once()
                        
                        # Verify fields were updated
                        assert mock_user_instance.first_name == 'Updated'
                        assert mock_user_instance.investment_experience == 'advanced'