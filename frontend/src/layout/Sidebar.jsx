import {
  LayoutDashboard,
  Upload,
  Video,
  FileText,
  Settings,
  LogOut,
} from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const Sidebar = () => {
  const navigate = useNavigate();

  const handleLogout = () => {
    // In a real app, clear tokens here
    navigate("/login");
  };

  const menuItems = [
    { name: "Dashboard", icon: LayoutDashboard, path: "/" },
    { name: "Upload Video", icon: Upload, path: "/upload-video" },
    { name: "Video Library", icon: Video, path: "/video-library" },
    { name: "Reports", icon: FileText, path: "/reports" },
    { name: "Settings", icon: Settings, path: "/settings" },
  ];

  return (
    <aside className="w-[280px] flex-shrink-0 min-h-screen bg-white/60 backdrop-blur-2xl border-r border-white/50 flex flex-col z-30 relative animate-slide-in-right shadow-[4px_0_24px_rgba(0,0,0,0.02)]">

      {/* Logo Section */}
      <div className="px-6 pt-8 pb-6">

        <motion.div 
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="flex items-center gap-3.5 group cursor-pointer" 
          onClick={() => navigate('/')}
        >

          {/* Premium Traffic Light */}
          <div className="relative w-12 h-12 bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl flex items-center justify-center shadow-lg shadow-slate-900/20 group-hover:shadow-indigo-500/20 group-hover:scale-105 transition-all duration-300">
            <div className="flex gap-1.5">
              <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse-glow shadow-[0_0_8px_rgba(244,63,94,0.6)]"></div>
              <div className="w-2 h-2 rounded-full bg-amber-400"></div>
              <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
            </div>
            <div className="absolute inset-0 rounded-2xl ring-1 ring-white/10 pointer-events-none"></div>
          </div>

          <div>
            <h1 className="text-2xl font-extrabold text-gradient-primary tracking-tight leading-none group-hover:opacity-80 transition-opacity">
              TrafficGuard
            </h1>
          </div>

        </motion.div>

      </div>

      {/* Gradient Divider */}
      <div className="gradient-divider mx-6 mb-2"></div>

      {/* Menu */}
      <div className="flex-1 px-4 pt-2 space-y-1 overflow-y-auto">

        {menuItems.map((item, index) => {
          const Icon = item.icon;

          return (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) => {
                const baseClasses = "flex items-center gap-3.5 px-4 py-3.5 rounded-xl transition-all duration-200 font-semibold text-[14px] group relative overflow-hidden";
                return isActive 
                  ? `${baseClasses} text-white shadow-md shadow-indigo-500/25`
                  : `${baseClasses} text-slate-500 hover:bg-slate-100/50 hover:text-slate-800`;
              }}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.div 
                      layoutId="active-nav"
                      className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-indigo-500 pointer-events-none z-0 rounded-xl"
                      transition={{ type: "spring", stiffness: 350, damping: 30 }}
                    />
                  )}
                  <motion.div
                    whileHover={!isActive ? { rotate: -8, scale: 1.15 } : {}}
                    transition={{ type: "spring", stiffness: 400, damping: 15 }}
                  >
                    <Icon size={20} className={`relative z-10 ${isActive ? "text-white" : "text-slate-400 group-hover:text-indigo-500 transition-colors"}`} strokeWidth={isActive ? 2.5 : 2} />
                  </motion.div>
                  <span className="relative z-10">
                    {item.name}
                  </span>
                  {isActive && (
                    <div className="absolute right-3 w-1.5 h-1.5 rounded-full bg-white/60 z-10"></div>
                  )}
                </>
              )}
            </NavLink>
          );
        })}

      </div>

      {/* Footer: Version + Logout */}
      <div className="px-4 pb-4 space-y-3">

        {/* Version badge */}
        <div className="gradient-divider mx-2 mb-1"></div>
        <div className="flex items-center justify-center gap-2 py-2">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">v2.4.1</span>
          <span className="w-1 h-1 rounded-full bg-emerald-500"></span>
          <span className="text-[10px] font-semibold text-emerald-500 uppercase tracking-wider">Stable</span>
        </div>

        <motion.button 
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleLogout}
          className="w-full flex items-center justify-center gap-2.5 bg-slate-100 hover:bg-rose-50 hover:text-rose-600 hover:border-rose-200 text-slate-600 border border-slate-200 rounded-xl px-4 py-3.5 transition-all duration-200 font-semibold text-sm cursor-pointer group"
        >

          <LogOut size={18} className="group-hover:text-rose-500 transition-colors" />

          <span>
            Sign Out
          </span>

        </motion.button>

      </div>

    </aside>
  );
};

export default Sidebar;