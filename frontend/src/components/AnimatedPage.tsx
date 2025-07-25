import { FC, ReactNode } from 'react';
import { motion } from 'framer-motion';

const pageVariants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
};

interface AnimatedPageProps {
  children: ReactNode;
  className?: string;
}

const AnimatedPage: FC<AnimatedPageProps> = ({ children, className }) => (
  <motion.div
    variants={pageVariants}
    initial="initial"
    animate="animate"
    exit="exit"
    transition={{ duration: 0.25, ease: 'easeInOut' }}
    className={className}
  >
    {children}
  </motion.div>
);

export default AnimatedPage;