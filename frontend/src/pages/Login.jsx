import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mail, Lock, ArrowRight, Eye, EyeOff, ShieldCheck } from 'lucide-react';

const Login = () => {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [focusedInput, setFocusedInput] = useState(null);

  const [errorMessage, setErrorMessage] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage("");
    
    try {
      const email = e.target.email.value;
      const password = e.target.password.value;
      
      const response = await fetch("http://localhost:8000/api/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || "Login failed");
      }
      
      // Store token (dummy for now)
      localStorage.setItem("token", data.token);
      
      navigate('/');
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 relative overflow-hidden font-sans p-4">
      {/* Premium Animated Background Mesh */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-indigo-500/20 rounded-full blur-[120px] animate-float pointer-events-none"></div>
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-500/20 rounded-full blur-[120px] animate-float delay-500 pointer-events-none"></div>
      <div className="absolute top-[30%] right-[20%] w-[30%] h-[30%] bg-rose-400/10 rounded-full blur-[100px] animate-pulse-glow pointer-events-none"></div>

      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9InJnYmEoMCwwLDAsMC4wNSkiLz48L3N2Zz4=')] [mask-image:linear-gradient(to_bottom,white,transparent)] pointer-events-none opacity-60"></div>

      <div className="relative z-10 w-full max-w-md animate-fade-in-up">
        
        {/* Main Card */}
        <div className="bg-white/80 backdrop-blur-2xl p-8 sm:p-10 rounded-[2.5rem] shadow-[0_20px_40px_-15px_rgba(0,0,0,0.1)] border border-white/60 relative overflow-hidden group">
          
          {/* Subtle top glare */}
          <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white to-transparent"></div>

          <div className="flex flex-col items-center text-center mb-10">
            <div className="relative w-16 h-16 bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl flex items-center justify-center shadow-xl shadow-slate-900/20 mb-6 group-hover:scale-105 transition-transform duration-500">
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse-glow shadow-[0_0_10px_rgba(244,63,94,0.6)]"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-amber-400"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
              </div>
              <div className="absolute inset-0 rounded-2xl ring-1 ring-white/10 pointer-events-none"></div>
            </div>
            
            <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight leading-none mb-2">
              TrafficGuard
            </h1>
            <div className="flex items-center gap-1.5 justify-center bg-indigo-50 text-indigo-600 px-3 py-1 rounded-full border border-indigo-100">
              <ShieldCheck size={14} />
              <p className="text-[10px] font-bold uppercase tracking-widest">
                Secure Admin Portal
              </p>
            </div>
          </div>

          <div className="mb-8">
            <h2 className="text-2xl font-bold text-slate-800 tracking-tight">
              Welcome back
            </h2>
            <p className="mt-1.5 text-sm text-slate-500 font-medium">
              Enter your credentials to access the system.
            </p>
          </div>

          {errorMessage && (
            <div className="mb-6 p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-start gap-3">
              <div className="mt-0.5">
                <svg className="h-5 w-5 text-rose-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-sm text-rose-600 font-medium leading-relaxed">{errorMessage}</p>
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-5">
            <div className="space-y-1.5">
              <label htmlFor="email" className="block text-sm font-semibold text-slate-700 ml-1">
                Email address
              </label>
              <div className="relative rounded-2xl shadow-sm transition-all duration-300">
                <div className={`absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none transition-colors ${focusedInput === 'email' ? 'text-indigo-600' : 'text-slate-400'}`}>
                  <Mail className="h-5 w-5" />
                </div>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  defaultValue="admin@trafficguard.com"
                  onFocus={() => setFocusedInput('email')}
                  onBlur={() => setFocusedInput(null)}
                  className="block w-full pl-12 pr-4 py-3.5 bg-slate-50/50 border border-slate-200 hover:border-slate-300 rounded-2xl text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all sm:text-sm font-medium"
                  placeholder="name@company.com"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label htmlFor="password" className="block text-sm font-semibold text-slate-700 ml-1">
                Password
              </label>
              <div className="relative rounded-2xl shadow-sm transition-all duration-300">
                <div className={`absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none transition-colors ${focusedInput === 'password' ? 'text-indigo-600' : 'text-slate-400'}`}>
                  <Lock className="h-5 w-5" />
                </div>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  required
                  defaultValue="password123"
                  onFocus={() => setFocusedInput('password')}
                  onBlur={() => setFocusedInput(null)}
                  className="block w-full pl-12 pr-12 py-3.5 bg-slate-50/50 border border-slate-200 hover:border-slate-300 rounded-2xl text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all sm:text-sm font-medium"
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-4 flex items-center text-slate-400 hover:text-indigo-600 cursor-pointer transition-colors"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center group/checkbox cursor-pointer">
                <div className="relative flex items-center justify-center w-5 h-5">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    defaultChecked
                    className="peer appearance-none w-5 h-5 border-2 border-slate-300 rounded-[6px] checked:bg-indigo-600 checked:border-indigo-600 cursor-pointer transition-all hover:border-indigo-500"
                  />
                  <ShieldCheck size={14} className="absolute text-white pointer-events-none opacity-0 peer-checked:opacity-100 transition-opacity" />
                </div>
                <label htmlFor="remember-me" className="ml-2.5 block text-sm text-slate-600 font-semibold cursor-pointer group-hover/checkbox:text-slate-800 transition-colors">
                  Remember me
                </label>
              </div>

              <div className="text-sm">
                <a href="#" className="font-bold text-indigo-600 hover:text-indigo-700 transition-colors">
                  Forgot password?
                </a>
              </div>
            </div>

            <div className="pt-4">
              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex justify-center py-4 px-4 border border-transparent rounded-2xl shadow-lg shadow-indigo-500/30 text-sm font-bold text-white bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/30 transition-all hover-lift disabled:opacity-70 disabled:cursor-not-allowed cursor-pointer group relative overflow-hidden"
              >
                {/* Shine effect */}
                <div className="absolute top-0 -inset-full h-full w-1/2 z-5 block transform -skew-x-12 bg-gradient-to-r from-transparent to-white opacity-20 group-hover:animate-shine"></div>

                {isLoading ? (
                  <span className="flex items-center gap-2 relative z-10">
                    <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Signing in...
                  </span>
                ) : (
                  <span className="flex items-center gap-2 relative z-10">
                    Sign in to Dashboard <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
                  </span>
                )}
              </button>
            </div>
          </form>
        </div>
        
        <p className="text-center text-sm font-semibold text-slate-400 mt-8">
          &copy; {new Date().getFullYear()} TrafficGuard Systems. All rights reserved.
        </p>
      </div>
    </div>
  );
};

export default Login;