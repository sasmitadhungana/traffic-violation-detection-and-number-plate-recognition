import { Routes, Route } from "react-router-dom";

import MainLayout from "../layout/MainLayout";

import Login from "../pages/Login";
import Dashboard from "../pages/Dashboard";
import UploadVideo from "../pages/UploadVideo";
import VideoLibrary from "../pages/VideoLibrary";
import Reports from "../pages/Reports";
import Settings from "../pages/Settings";

const AppRoutes = () => {
  return (
    <Routes>

      <Route path="/login" element={<Login />} />

      <Route element={<MainLayout />}>

        <Route path="/" element={<Dashboard />} />

        <Route
          path="/upload-video"
          element={<UploadVideo />}
        />

        <Route
          path="/video-library"
          element={<VideoLibrary />}
        />

        <Route
          path="/reports"
          element={<Reports />}
        />

        <Route
          path="/settings"
          element={<Settings />}
        />

      </Route>

    </Routes>
  );
};

export default AppRoutes;