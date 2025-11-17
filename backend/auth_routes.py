from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from pydantic import BaseModel
from typing import Optional

from auth import authenticate_user, create_access_token, verify_token
from config import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/auth", tags=["authentication"])

# Request/Response Models
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login", response_model=Token)
async def login_for_access_token(login_data: LoginRequest):
    """Login endpoint - returns JWT token"""
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register")
async def register_user(user_data: UserCreate):
    """Register a new user"""
    # In a real implementation, save to database
    # For demo, just return success
    return {
        "message": "User registered successfully",
        "username": user_data.username,
        "email": user_data.email
    }

@router.get("/me", response_model=User)
async def read_users_me(current_username: str = Depends(verify_token)):
    """Get current user information"""
    # In a real implementation, fetch from database
    user = {
        "username": current_username,
        "email": "demo@example.com",
        "full_name": "Demo User",
        "disabled": False
    }
    return User(**user)

@router.post("/logout")
async def logout_user(current_username: str = Depends(verify_token)):
    """Logout endpoint (token revocation would be implemented here)"""
    return {"message": "Successfully logged out"}

@router.get("/verify")
async def verify_token_endpoint(token: str):
    """Verify if a token is valid"""
    try:
        # Simple token verification (in production, use proper validation)
        if len(token) > 50:  # Basic JWT length check
            return {"valid": True, "message": "Token is valid"}
        else:
            return {"valid": False, "message": "Invalid token"}
    except Exception:
        return {"valid": False, "message": "Token verification failed"}