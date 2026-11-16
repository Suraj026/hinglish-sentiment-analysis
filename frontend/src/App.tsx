import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { AnalyzeText } from './pages/AnalyzeText';
import { ModelLab } from './pages/ModelLab';

export const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="analyze" element={<AnalyzeText />} />
        <Route path="models" element={<ModelLab />} />
        <Route path="benchmarks" element={<Navigate to="/models" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
};

export default App;
