"use client";

import { Activity, Stethoscope, FileText, Settings } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const Sidebar = () => {
    const pathname = usePathname();

    return (
        <aside style={{
            width: '260px',
            backgroundColor: '#ffffff',
            borderRight: '1px solid var(--border)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            height: '100vh',
            position: 'sticky',
            top: 0
        }}>
            <div style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--primary-color)' }}>
                <Activity size={32} />
                <h1 style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--text-main)' }}>MediScan AI</h1>
            </div>

            <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1 }}>
                <p style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 'bold', marginBottom: '0.5rem' }}>Diagnosis Modules</p>

                <Link href="/" style={{ textDecoration: 'none' }}>
                    <NavItem icon={<Stethoscope size={20} />} label="Pneumonia" active={pathname === '/'} />
                </Link>

                <Link href="/tuberculosis" style={{ textDecoration: 'none' }}>
                    <NavItem icon={<FileText size={20} />} label="Tuberculosis" active={pathname === '/tuberculosis'} />
                </Link>

                <NavItem icon={<Activity size={20} />} label="COVID-19" disabled />
            </nav>

            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                <NavItem icon={<Settings size={20} />} label="Settings" />
            </div>
        </aside>
    );
};

const NavItem = ({ icon, label, active = false, disabled = false }: { icon: any, label: string, active?: boolean, disabled?: boolean }) => {
    return (
        <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '0.75rem 1rem',
            borderRadius: 'var(--radius)',
            backgroundColor: active ? 'var(--primary-color)' : 'transparent',
            color: active ? 'white' : disabled ? 'var(--text-muted)' : 'var(--text-main)',
            cursor: disabled ? 'not-allowed' : 'pointer',
            opacity: disabled ? 0.6 : 1,
            transition: 'all 0.2s',
            fontWeight: 500
        }}>
            {icon}
            <span>{label}</span>
            {disabled && <span style={{ marginLeft: 'auto', fontSize: '0.65rem', backgroundColor: '#e2e8f0', padding: '2px 6px', borderRadius: '4px' }}>SOON</span>}
        </div>
    );
};

export default Sidebar;
